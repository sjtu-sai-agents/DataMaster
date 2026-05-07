"""Feishu document writer

Wraps Feishu docx.v1 API, providing a concise interface for creating documents and appending content blocks.
Used to write the full Agent trajectory to a Feishu document (no truncation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lark_oapi as lark

from lark_oapi.api.docx.v1 import (
    BatchDeleteDocumentBlockChildrenRequest,
    BatchDeleteDocumentBlockChildrenRequestBody,
    Block,
    CreateDocumentBlockChildrenRequest,
    CreateDocumentBlockChildrenRequestBody,
    CreateDocumentRequest,
    CreateDocumentRequestBody,
    Divider,
    ListDocumentBlockRequest,
    PatchDocumentBlockRequest,
    Text,
    TextElement,
    TextElementStyle,
    TextRun,
    TextStyle,
    UpdateBlockRequest,
    UpdateTextElementsRequest,
)
from lark_oapi.api.drive.v1 import (
    Owner,
    PatchPermissionPublicRequest,
    PermissionPublic,
    PermissionPublicRequest,
    TransferOwnerPermissionMemberRequest,
)

logger = logging.getLogger(__name__)

# Block type constants (Feishu docx.v1)
_BT_TEXT = 2
_BT_HEADING1 = 3
_BT_HEADING2 = 4
_BT_HEADING3 = 5
_BT_HEADING4 = 6
_BT_CODE = 14
_BT_DIVIDER = 22
_BT_IMAGE = 27

# Code language: 1 = PlainText, 49 = JSON, 15 = Python
_LANG_PLAIN = 1
_LANG_JSON = 49
_LANG_PYTHON = 15

# Max chars per code block (Feishu limit is ~64KB per block)
_MAX_CODE_BLOCK_CHARS = 30000

_LANG_MAP = {
    "plaintext": _LANG_PLAIN,
    "json": _LANG_JSON,
    "python": _LANG_PYTHON,
}

# Block type → human-readable name (for list_blocks output)
_BLOCK_TYPE_NAMES = {
    1: "page", 2: "text",
    3: "heading1", 4: "heading2", 5: "heading3", 6: "heading4",
    7: "heading5", 8: "heading6", 9: "heading7", 10: "heading8", 11: "heading9",
    14: "code", 22: "divider", 27: "image",
}


class FeishuDocumentWriter:
    """Feishu document writer: create documents and append content blocks."""

    def __init__(
        self,
        client: lark.Client,
        folder_token: str | None = None,
        domain: str = "https://open.feishu.cn",
    ):
        """Initialize the document writer.

        Args:
            client: Feishu Client instance.
            folder_token: Optional folder token for document placement.
            domain: Feishu API domain.
        """
        self._client = client
        self._folder_token = folder_token
        # Extract base domain for URL construction
        # "https://open.feishu.cn" -> "feishu.cn"
        # "https://open.larksuite.com" -> "larksuite.com"
        host = domain.replace("https://", "").replace("http://", "")
        host = host.removeprefix("open.")
        self._base_domain = host

    def create_document(self, title: str) -> str | None:
        """Create a Feishu document.

        Returns:
            document_id on success, None on failure.
        """
        body_builder = CreateDocumentRequestBody.builder().title(title[:800])
        if self._folder_token:
            body_builder = body_builder.folder_token(self._folder_token)

        request = (
            CreateDocumentRequest.builder()
            .request_body(body_builder.build())
            .build()
        )

        try:
            response = self._client.docx.v1.document.create(request)
            if not response.success():
                logger.warning(
                    "Failed to create document: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return None
            doc_id = response.data.document.document_id
            logger.info("Created Feishu document: %s", doc_id)
            return doc_id
        except Exception:
            logger.exception("Exception creating document")
            return None

    def set_public_readable(self, document_id: str) -> bool:
        """Set the document to link-readable permission."""
        request = (
            PatchPermissionPublicRequest.builder()
            .token(document_id)
            .type("docx")
            .request_body(
                PermissionPublicRequest.builder()
                .link_share_entity("anyone_readable")
                .external_access(True)
                .build()
            )
            .build()
        )

        try:
            response = self._client.drive.v1.permission_public.patch(request)
            if not response.success():
                logger.warning(
                    "Failed to set doc permission: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return False
            return True
        except Exception:
            logger.exception("Exception setting document permission")
            return False

    def get_document_url(self, document_id: str) -> str:
        """Build the document URL."""
        return f"https://{self._base_domain}/docx/{document_id}"

    def transfer_ownership(self, document_id: str, user_open_id: str) -> bool:
        """Transfer document ownership to the specified user.

        After transfer, the user can manage (edit, delete) the document themselves.

        Args:
            document_id: Document ID.
            user_open_id: Target user's open_id.
        """
        request = (
            TransferOwnerPermissionMemberRequest.builder()
            .token(document_id)
            .type("docx")
            .need_notification(False)
            .request_body(
                Owner.builder()
                .member_type("openid")
                .member_id(user_open_id)
                .build()
            )
            .build()
        )

        try:
            response = self._client.drive.v1.permission_member.transfer_owner(request)
            if not response.success():
                logger.warning(
                    "Failed to transfer doc ownership: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return False
            logger.info(
                "Transferred doc %s ownership to %s", document_id, user_open_id
            )
            return True
        except Exception:
            logger.exception("Exception transferring document ownership")
            return False

    def append_blocks(self, document_id: str, blocks: list[Block]) -> bool:
        """Batch append blocks to the end of the document."""
        if not blocks:
            return True

        request = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(document_id)
            .block_id(document_id)  # root block = document_id
            .request_body(
                CreateDocumentBlockChildrenRequestBody.builder()
                .children(blocks)
                .build()
            )
            .build()
        )

        try:
            response = self._client.docx.v1.document_block_children.create(request)
            if not response.success():
                logger.warning(
                    "Failed to append blocks: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return False
            return True
        except Exception:
            logger.exception("Exception appending blocks to document %s", document_id)
            return False

    # ---- Convenience methods ----

    def append_heading(
        self, document_id: str, text: str, level: int = 3
    ) -> bool:
        """Append a heading block (level 1-9)."""
        block = _build_heading_block(text, level)
        return self.append_blocks(document_id, [block])

    def append_text(
        self, document_id: str, text: str, bold: bool = False
    ) -> bool:
        """Append a text paragraph."""
        block = _build_text_block(text, bold=bold)
        return self.append_blocks(document_id, [block])

    def append_code_block(
        self, document_id: str, code: str, language: str = "plaintext"
    ) -> bool:
        """Append a code block."""
        if len(code) > _MAX_CODE_BLOCK_CHARS:
            code = code[:_MAX_CODE_BLOCK_CHARS] + "\n... (content truncated)"
        block = _build_code_block(code, language)
        return self.append_blocks(document_id, [block])

    def append_divider(self, document_id: str) -> bool:
        """Append a divider line."""
        block = _build_divider_block()
        return self.append_blocks(document_id, [block])

    def append_image(self, document_id: str, file_token: str) -> bool:
        """Append an image block.

        Args:
            document_id: Document ID.
            file_token: Uploaded image file_token (obtained from upload_media_for_doc).
        """
        block = _build_image_block(file_token)
        return self.append_blocks(document_id, [block])

    def insert_image(
        self, document_id: str, file_token: str, index: int
    ) -> bool:
        """Insert an image block at the specified position."""
        block = _build_image_block(file_token)
        return self.insert_blocks(document_id, [block], index)

    def upload_and_append_image(self, document_id: str, file_path: str) -> str | None:
        """Upload an image and append it to the end of the document.

        Args:
            document_id: Document ID.
            file_path: Local image file path.

        Returns:
            file_token on success, None on failure.
        """
        file_token = upload_media_for_doc(self._client, file_path, document_id)
        if not file_token:
            return None
        ok = self.append_image(document_id, file_token)
        if not ok:
            logger.warning("Image uploaded (token=%s) but failed to append to doc", file_token)
        return file_token

    # ---- Block editing methods ----

    def list_blocks(self, document_id: str) -> list[dict] | None:
        """List all blocks in a document and return a simplified structure list.

        Returns:
            A list of dicts: {index, block_id, block_type, block_type_name, text_content}
            Returns None on failure.
        """
        all_blocks: list = []
        page_token = None

        while True:
            builder = (
                ListDocumentBlockRequest.builder()
                .document_id(document_id)
                .page_size(500)
            )
            if page_token:
                builder = builder.page_token(page_token)

            request = builder.build()
            try:
                response = self._client.docx.v1.document_block.list(request)
                if not response.success():
                    logger.warning(
                        "Failed to list blocks: code=%s, msg=%s",
                        response.code, response.msg,
                    )
                    return None

                items = response.data.items or []
                all_blocks.extend(items)

                if not response.data.has_more:
                    break
                page_token = response.data.page_token
            except Exception:
                logger.exception("Exception listing blocks for document %s", document_id)
                return None

        result = []
        for idx, block in enumerate(all_blocks):
            block_type = block.block_type
            result.append({
                "index": idx,
                "block_id": block.block_id,
                "block_type": block_type,
                "block_type_name": _BLOCK_TYPE_NAMES.get(block_type, f"unknown({block_type})"),
                "text_content": _extract_block_text(block),
            })
        return result

    def update_block_text(
        self, document_id: str, block_id: str, new_text: str
    ) -> bool:
        """Update the text content of a specified block (replaces all TextElements).

        Applicable to text, heading, and code block types.
        """
        request = (
            PatchDocumentBlockRequest.builder()
            .document_id(document_id)
            .block_id(block_id)
            .document_revision_id(-1)
            .request_body(
                UpdateBlockRequest.builder()
                .update_text_elements(
                    UpdateTextElementsRequest.builder()
                    .elements([_build_text_run(new_text)])
                    .build()
                )
                .build()
            )
            .build()
        )

        try:
            response = self._client.docx.v1.document_block.patch(request)
            if not response.success():
                logger.warning(
                    "Failed to update block %s: code=%s, msg=%s",
                    block_id, response.code, response.msg,
                )
                return False
            return True
        except Exception:
            logger.exception(
                "Exception updating block %s in document %s", block_id, document_id
            )
            return False

    def delete_blocks(
        self, document_id: str, start_index: int, end_index: int
    ) -> bool:
        """Delete child blocks in the range [start_index, end_index) under the document root block."""
        request = (
            BatchDeleteDocumentBlockChildrenRequest.builder()
            .document_id(document_id)
            .block_id(document_id)
            .document_revision_id(-1)
            .request_body(
                BatchDeleteDocumentBlockChildrenRequestBody.builder()
                .start_index(start_index)
                .end_index(end_index)
                .build()
            )
            .build()
        )

        try:
            response = self._client.docx.v1.document_block_children.batch_delete(request)
            if not response.success():
                logger.warning(
                    "Failed to delete blocks [%d, %d): code=%s, msg=%s",
                    start_index, end_index, response.code, response.msg,
                )
                return False
            return True
        except Exception:
            logger.exception(
                "Exception deleting blocks [%d, %d) in document %s",
                start_index, end_index, document_id,
            )
            return False

    def insert_blocks(
        self, document_id: str, blocks: list[Block], index: int
    ) -> bool:
        """Insert blocks at the specified position in the document."""
        if not blocks:
            return True

        request = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(document_id)
            .block_id(document_id)
            .request_body(
                CreateDocumentBlockChildrenRequestBody.builder()
                .children(blocks)
                .index(index)
                .build()
            )
            .build()
        )

        try:
            response = self._client.docx.v1.document_block_children.create(request)
            if not response.success():
                logger.warning(
                    "Failed to insert blocks at index %d: code=%s, msg=%s",
                    index, response.code, response.msg,
                )
                return False
            return True
        except Exception:
            logger.exception(
                "Exception inserting blocks at index %d in document %s",
                index, document_id,
            )
            return False

    # ---- Insert convenience methods ----

    def insert_heading(
        self, document_id: str, text: str, index: int, level: int = 3
    ) -> bool:
        """Insert a heading block at the specified position."""
        block = _build_heading_block(text, level)
        return self.insert_blocks(document_id, [block], index)

    def insert_text(
        self, document_id: str, text: str, index: int, bold: bool = False
    ) -> bool:
        """Insert a text paragraph at the specified position."""
        block = _build_text_block(text, bold=bold)
        return self.insert_blocks(document_id, [block], index)

    def insert_code_block(
        self, document_id: str, code: str, index: int, language: str = "plaintext"
    ) -> bool:
        """Insert a code block at the specified position."""
        if len(code) > _MAX_CODE_BLOCK_CHARS:
            code = code[:_MAX_CODE_BLOCK_CHARS] + "\n... (content truncated)"
        block = _build_code_block(code, language)
        return self.insert_blocks(document_id, [block], index)

    def insert_divider(self, document_id: str, index: int) -> bool:
        """Insert a divider line at the specified position."""
        block = _build_divider_block()
        return self.insert_blocks(document_id, [block], index)


# ---- Block builder helpers ----

def _build_text_run(content: str, bold: bool = False) -> TextElement:
    """Build a TextElement (TextRun)."""
    style_builder = TextElementStyle.builder()
    if bold:
        style_builder = style_builder.bold(True)

    return (
        TextElement.builder()
        .text_run(
            TextRun.builder()
            .content(content)
            .text_element_style(style_builder.build())
            .build()
        )
        .build()
    )


def _build_text_block(content: str, bold: bool = False) -> Block:
    """Build a text paragraph Block."""
    return (
        Block.builder()
        .block_type(_BT_TEXT)
        .text(
            Text.builder()
            .elements([_build_text_run(content, bold=bold)])
            .build()
        )
        .build()
    )


def _build_heading_block(content: str, level: int = 3) -> Block:
    """Build a heading Block (level 1-9)."""
    level = max(1, min(9, level))
    block_type = _BT_HEADING1 + level - 1  # heading1=3, heading2=4, ...

    heading_text = (
        Text.builder()
        .elements([_build_text_run(content)])
        .build()
    )

    builder = Block.builder().block_type(block_type)
    # Map level to the correct heading method
    heading_setter = getattr(builder, f"heading{level}", None)
    if heading_setter:
        builder = heading_setter(heading_text)
    else:
        builder = builder.heading3(heading_text)

    return builder.build()


def _build_code_block(code: str, language: str = "plaintext") -> Block:
    """Build a code block Block."""
    lang_id = _LANG_MAP.get(language, _LANG_PLAIN)

    return (
        Block.builder()
        .block_type(_BT_CODE)
        .code(
            Text.builder()
            .elements([_build_text_run(code)])
            .style(TextStyle.builder().language(lang_id).build())
            .build()
        )
        .build()
    )


def _build_divider_block() -> Block:
    """Build a divider Block."""
    return Block.builder().block_type(_BT_DIVIDER).divider(Divider.builder().build()).build()


def _build_image_block(file_token: str) -> Block:
    """Build an image Block.

    Args:
        file_token: The file_token obtained from upload_media_for_doc.
    """
    from lark_oapi.api.docx.v1 import Image as DocImage

    return (
        Block.builder()
        .block_type(_BT_IMAGE)
        .image(DocImage.builder().token(file_token).build())
        .build()
    )


def upload_media_for_doc(client, file_path: str, doc_id: str) -> str | None:
    """Upload a file to Feishu Drive (for document images) and return the file_token.

    Args:
        client: Feishu Client instance.
        file_path: Local file path.
        doc_id: Target document ID.

    Returns:
        file_token on success, None on failure.
    """
    from lark_oapi.api.drive.v1 import UploadAllMediaRequest, UploadAllMediaRequestBody

    p = Path(file_path)
    if not p.exists():
        logger.error("File not found for doc upload: %s", file_path)
        return None

    file_size = p.stat().st_size
    file_name = p.name

    try:
        with open(file_path, "rb") as f:
            request = (
                UploadAllMediaRequest.builder()
                .request_body(
                    UploadAllMediaRequestBody.builder()
                    .file_name(file_name)
                    .parent_type("docx_image")
                    .parent_node(doc_id)
                    .size(file_size)
                    .file(f)
                    .build()
                )
                .build()
            )
            response = client.drive.v1.media.upload_all(request)

        if not response.success():
            logger.error(
                "Failed to upload media for doc: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        file_token = response.data.file_token
        logger.debug("Media uploaded for doc: %s -> %s", file_path, file_token)
        return file_token

    except Exception:
        logger.exception("Error uploading media %s for document %s", file_path, doc_id)
        return None


def _extract_block_text(block) -> str:
    """Extract plain text content from a Block object."""
    text_obj = getattr(block, "text", None)
    if text_obj is None:
        for level in range(1, 10):
            text_obj = getattr(block, f"heading{level}", None)
            if text_obj is not None:
                break
    if text_obj is None:
        text_obj = getattr(block, "code", None)
    if text_obj is None:
        return ""

    elements = getattr(text_obj, "elements", None)
    if not elements:
        return ""

    parts = []
    for elem in elements:
        text_run = getattr(elem, "text_run", None)
        if text_run:
            content = getattr(text_run, "content", "")
            if content:
                parts.append(content)
    return "".join(parts)
