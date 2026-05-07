import os
import numpy as np
import pandas as pd


def add_features(df):
    """Feature engineering function for semiconductor properties prediction."""
    df = df.copy()
    # Oxygen percentage
    df['percent_atom_o'] = 1.0 - (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in'])
    # Lattice angles to radians
    df['lattice_angle_alpha_rad'] = np.radians(df['lattice_angle_alpha_degree'])
    df['lattice_angle_beta_rad'] = np.radians(df['lattice_angle_beta_degree'])
    df['lattice_angle_gamma_rad'] = np.radians(df['lattice_angle_gamma_degree'])
    # Cell volume (triclinic formula)
    a = df['lattice_vector_1_ang']
    b = df['lattice_vector_2_ang']
    c = df['lattice_vector_3_ang']
    cos_alpha = np.cos(df['lattice_angle_alpha_rad'])
    cos_beta = np.cos(df['lattice_angle_beta_rad'])
    cos_gamma = np.cos(df['lattice_angle_gamma_rad'])
    term = 1 + 2*cos_alpha*cos_beta*cos_gamma - cos_alpha**2 - cos_beta**2 - cos_gamma**2
    term = np.clip(term, 0, None)  # numerical safety
    df['cell_volume'] = a * b * c * np.sqrt(term)
    # Volume per atom
    df['volume_per_atom'] = df['cell_volume'] / df['number_of_total_atoms']
    # Squared fractions
    df['al_frac_sq'] = df['percent_atom_al'] ** 2
    df['ga_frac_sq'] = df['percent_atom_ga'] ** 2
    df['in_frac_sq'] = df['percent_atom_in'] ** 2
    df['o_frac_sq'] = df['percent_atom_o'] ** 2
    # Interaction with total atoms
    df['al_by_atoms'] = df['percent_atom_al'] * df['number_of_total_atoms']
    df['ga_by_atoms'] = df['percent_atom_ga'] * df['number_of_total_atoms']
    df['in_by_atoms'] = df['percent_atom_in'] * df['number_of_total_atoms']
    df['o_by_atoms'] = df['percent_atom_o'] * df['number_of_total_atoms']
    # Metal ratios (add epsilon to avoid division by zero)
    eps = 1e-6
    df['al_over_ga'] = df['percent_atom_al'] / (df['percent_atom_ga'] + eps)
    df['al_over_in'] = df['percent_atom_al'] / (df['percent_atom_in'] + eps)
    df['ga_over_in'] = df['percent_atom_ga'] / (df['percent_atom_in'] + eps)
    # Categorical features
    df['spacegroup'] = df['spacegroup'].astype('category')
    df['number_of_total_atoms'] = df['number_of_total_atoms'].astype('category')
    return df


class MyDataLoader(BaseDataLoader):
    """Data loader for semiconductor properties prediction."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dir = kwargs.get('input_dir', './input')
        
    def setup(self):
        """
        Load data, perform feature engineering, and split train/validation.
        Uses val.csv for validation if available, otherwise falls back to random split.
        """
        # Load data
        train_df = pd.read_csv(os.path.join(self.input_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(self.input_dir, "test.csv"))
        
        # Apply feature engineering
        train_fe = add_features(train_df)
        test_fe = add_features(test_df)
        
        # Define feature columns (exclude id and targets)
        exclude_cols = ['id', 'formation_energy_ev_natom', 'bandgap_energy_ev']
        feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
        
        # Prepare features and targets
        X = train_fe[feature_cols]
        y1 = np.log1p(train_fe['formation_energy_ev_natom'])
        y2 = np.log1p(train_fe['bandgap_energy_ev'])
        X_test = test_fe[feature_cols]
        test_ids = test_df['id']
        
        # Identify categorical columns for LightGBM
        cat_features = [col for col in feature_cols if train_fe[col].dtype.name == 'category']
        
        # Split train/val - check for val.csv first (CRITICAL: use fixed validation set)
        val_path = os.path.join(self.input_dir, "val.csv")
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            val_ids = set(val_df['id'].values)
            
            # Check if val.csv has target columns
            if 'formation_energy_ev_natom' in val_df.columns:
                # val.csv has full data with targets
                val_fe = add_features(val_df)
                X_val = val_fe[feature_cols].reset_index(drop=True)
                y1_val = np.log1p(val_fe['formation_energy_ev_natom']).reset_index(drop=True)
                y2_val = np.log1p(val_fe['bandgap_energy_ev']).reset_index(drop=True)
            else:
                # val.csv has only ids, extract validation data from train
                val_mask = train_fe['id'].isin(val_ids)
                X_val = X.loc[val_mask].reset_index(drop=True)
                y1_val = y1.loc[val_mask].reset_index(drop=True)
                y2_val = y2.loc[val_mask].reset_index(drop=True)
            
            # Remove validation samples from training set
            train_mask = ~train_fe['id'].isin(val_ids)
            X_train = X.loc[train_mask].reset_index(drop=True)
            y1_train = y1.loc[train_mask].reset_index(drop=True)
            y2_train = y2.loc[train_mask].reset_index(drop=True)
        else:
            # Fallback to random split only if val.csv doesn't exist
            from sklearn.model_selection import train_test_split
            X_train, X_val, y1_train, y1_val = train_test_split(
                X, y1, test_size=0.2, random_state=42
            )
            _, _, y2_train, y2_val = train_test_split(
                X, y2, test_size=0.2, random_state=42
            )
        
        # Set train_data and test_data
        self.train_data = {
            'X_train': X_train,
            'X_val': X_val,
            'y1_train': y1_train,
            'y1_val': y1_val,
            'y2_train': y2_train,
            'y2_val': y2_val,
            'cat_features': cat_features,
            'feature_cols': feature_cols,
            'X_full': X,
            'y1_full': y1,
            'y2_full': y2
        }
        self.test_data = {
            'X_test': X_test,
            'test_ids': test_ids
        }
        
    def describe(self) -> str:
        """
        Return a description of the data processing approach.
        """
        return ("Data loader for semiconductor properties prediction. "
                "Features include: oxygen percentage, lattice angles (radians), "
                "cell volume (triclinic formula), volume per atom, squared fractions, "
                "metal ratios, and categorical encoding for spacegroup and number_of_total_atoms. "
                "Uses fixed validation set from input/val.csv if available.")