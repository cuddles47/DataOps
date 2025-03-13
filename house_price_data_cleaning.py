#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# ## Loading the data from separate files to combine in one 

# In[ ]:


# Load CSV files into individual Pandas DataFrames, setting `low_memory=False` to optimize memory usage 
data_ab = pd.read_csv('/kaggle/input/canada-housing/data_ab.csv', low_memory=False)
data_bc = pd.read_csv('/kaggle/input/canada-housing/data_bc.csv', low_memory=False)
data_mb = pd.read_csv('/kaggle/input/canada-housing/data_mb.csv', low_memory=False)
data_nb = pd.read_csv('/kaggle/input/canada-housing/data_nb.csv', low_memory=False)
data_nl = pd.read_csv('/kaggle/input/canada-housing/data_nl.csv', low_memory=False)
data_ns = pd.read_csv('/kaggle/input/canada-housing/data_ns.csv', low_memory=False)
data_nt = pd.read_csv('/kaggle/input/canada-housing/data_nt.csv', low_memory=False)
data_on = pd.read_csv('/kaggle/input/canada-housing/data_on.csv', low_memory=False)
data_pe = pd.read_csv('/kaggle/input/canada-housing/data_pe.csv', low_memory=False)
data_sk = pd.read_csv('/kaggle/input/canada-housing/data_sk.csv', low_memory=False)
data_yt = pd.read_csv('/kaggle/input/canada-housing/data_yt.csv', low_memory=False)


# In[ ]:


# Concatenate all DataFrames along the rows (`axis=0`) to create a unified dataset

df = pd.concat([data_ab, data_bc, data_mb, data_nb, data_nl, 
                data_ns, data_nt, data_on, data_pe, data_sk,
                data_yt], axis=0) 
df.shape


# The dataset is quite big. Let's drop null rows and columms

# In[ ]:


# Drop duplicated rows
df = df.drop_duplicates()

# Drop duplicate columns
df = df.T.drop_duplicates().T
df.shape


# With option pd.set_option('display.max_columns', None) we can see all features

# In[ ]:


pd.set_option('display.max_columns', None)
df.head()


# There is no need in data without address or price, so we can drop all rows with null values in "streetAddress", "addressLocality", "addressRegion", "price" columns:

# In[ ]:


df = df.dropna(subset=["streetAddress", "addressLocality", "addressRegion", "price"])


# In[ ]:


df.shape


# # Square Footage and property-sqft

# Columns 'Square Footage', 'property-sqft' have the same information. Let's combine them in one column. 

# In[ ]:


df[['Square Footage', 'property-sqft']].info()


# In[ ]:


df.loc[:, 'Square Footage new'] = df['Square Footage'].str.replace(' SQFT', '', regex=False)
df.loc[:, 'Square Footage new'] = df['Square Footage new'].fillna(df['property-sqft'])

df[['property-sqft', 'Square Footage', 'Square Footage new']].head()


# In[ ]:


df[['Square Footage new']].info()


# # Acreage

# In[ ]:


df.Acreage.head()


# "Acreage" typically refers to a large piece of land, usually measured in acres. In real estate, it implies that the property includes a significant amount of land, often used for farming, ranching, or simply as a large private estate. So, Nan values can be replaced to 0

# In[ ]:


df["Acreage"] = df["Acreage"].fillna(0)
df.Acreage.head()


# # property-bath

# In[ ]:


df[['property-baths', 'Bath']].describe()


# In[ ]:


# Get unique values from both columns
unique_values = pd.Index(df['property-baths'].dropna().unique()).union(df['Bath'].dropna().unique())

# Create a DataFrame with unique values as index
counts_df = pd.DataFrame(index=unique_values, columns=['property-baths', 'Bath'])

# Count occurrences and fill the DataFrame
counts_df['property-baths'] = df['property-baths'].value_counts().reindex(unique_values, fill_value=0)
counts_df['Bath'] = df['Bath'].value_counts().reindex(unique_values, fill_value=0)

print(counts_df)


# In[ ]:


pd.set_option('display.max_colwidth', None)

df[df['property-baths'] == 42.0][['property-baths', 'Bath', 'description', 'price']]


# In[ ]:


df[df['property-baths'] == 803][['property-baths', 'Bath', 'description', 'price']]


# In[ ]:


df.loc[df['property-baths'] == 803, 'property-baths'] = 1


# In[ ]:


df[['property-baths']].value_counts()


# # Features

# In[ ]:


df.Features.head()


# The column 'Features' contains detailed information about a house, including parking, garage, pool, heating, garden, and other. Some rows may have multiple features listed, while others might only contain a few or even none. Additionally, the 'Garage' column can sometimes be empty, even when garage-related details are present in the 'Features' column. This makes it crucial to extract and analyze all values from 'Features' to ensure no important information is missed.

# In[ ]:


df['Features'].unique().shape


# We have 5,338 different values in the 'Feature' column. The problem is that each cell can contain a list of information, with potential relationships between features in the dataset. We can define some keywords for each feature, extract relevant values, and fill in the gaps.

# In[ ]:


##### Extract unique features
def get_unique_values(column):

    unique_features = set()

    for features in df[column].dropna():
        # Remove brackets and extra quotes
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "")
        # Split by comma
        feature_list = [f.strip() for f in cleaned_features.split(",")]
        unique_features.update(feature_list)

    return unique_features


# In[ ]:


unique_features = get_unique_values('Features')
print(len(unique_features), unique_features)


# Now we got 268 different unique values. Let's fill the gap in Garage column

# ## Garage

# In[ ]:


df.Garage.value_counts()


# Column 'Garage' contains only 'Yes' or 'No' values. The information about type of garage, how namy of them in the house could contain columns 'Feature' or 'Parking'. Let's combine them.

# In[ ]:


unique_parking_features = get_unique_values('Parking') | unique_features
print(len(unique_parking_features), unique_parking_features)


# Let's select from this list only garage related terminology:

# In[ ]:


garage_keywords = {"garage", "carport", "grge", "attached", "detached", "tandem", "car"}

# The selected keywords cover different terms related to garages and carports:

#     "garage" → Captures any mention of a garage.
#     "carport" → Includes carports, which serve a similar function.
#     "grge" → A common abbreviation for "garage" (e.g., "DetachedGrge/Carport").
#     "attached" → Covers garages that are attached to the house.
#     "detached" → Covers garages that are separate from the house.
#     "tandem" → Represents tandem garages, where cars are parked one behind the other.
#     "car" → Captures features mentioning a car (e.g., "2 Car Attached", "3 Car Detached").

garage_list = []


for item in unique_parking_features:
    lower_item = item.lower()
    if any(keyword in lower_item for keyword in garage_keywords):
        garage_list.append(item.lower())

garage_list.remove('no garage')

print("Garage-related items:", garage_list)


# In[ ]:


def check_keywords(features, keywords):

    try:
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "").replace(";", ",")
        list_to_check = [f.strip().lower() for f in cleaned_features.split(',')]
        common = set(list_to_check) & set(keywords)

        if common: 
            return 'Yes'  

    except AttributeError:
        return 'No'

    return 'No'


# The function 'check_keywords' will check if column 'Features' and 'Parking' contain at least one pfrase from the list 'garage_list'. In this case, the nan value in column 'Garage' will be changes to "Yes".

# In[ ]:


df['Garage new'] = df['Garage']
df.loc[df.Garage.isna(), 'Garage new'] = df.loc[df.Garage.isna(), 'Features'].apply(lambda x: check_keywords(x, garage_list))
df['Garage new'].value_counts()


# Some rows can have information about garage in Parking column

# In[ ]:


df.Parking.unique()


# In[ ]:


df.loc[df.Garage == 'No', 'Garage new'] = df.loc[df.Garage == 'No', 'Parking'].apply(lambda x: check_keywords(x, garage_list))
df['Garage new'].value_counts()


# The number of not-null values does not changed. So all possible values were filled from 'Features' column.

# ## Let's do the same for column 'Parking'

# In[ ]:


df.Parking.value_counts()


# In[ ]:


parking_list = ["2 Outdoor Stalls", "Add. Parking Avail.", "Additional Parking", "Assigned",
                "Carport", "Carport & Garage", "Carport Double", "Carport Quad+", "Carport Triple", 
                "Carport; Multiple", "Carport; Single", "Multiple Driveways", "Parkade", "Parking Lot", 
                "Parking Pad", "Parking Pad Cement/Paved", "Parking Space(s)", "Parking Spaces", 
                "RV", "RV Access/Parking", "RV Gated", "RV Hookup", "RV Parking", "RV Parking Avail.", 
                "Shared Driveway", "Stall", "Tandem Parking", "Underground", "Underground Parking", 
                "Visitor Parking"
                ]

parking_list = [f.lower() for f in parking_list]


# We'll replace values in a Parking column to 'Yes' and "No'

# In[ ]:


# first, copy the column
df['Parking new'] = df['Parking']

# if the column 'Parking' has one of the words from the parking_list, it will be replaced to 'Yes'
df['Parking new'] = df['Parking new'].apply(lambda x: check_keywords(x, parking_list))

# if the column 'Features' has one of the words from the parking_list, it will be replaced to 'Yes' in column 'Parking'
df.loc[df['Parking new'] == 'No', 'Parking new'] = df.loc[df['Parking new']  == 'No', 'Features'].apply(lambda x: check_keywords(x, parking_list))

df['Parking new'].value_counts()


# ## Basement

# In[ ]:


df.Basement.value_counts()


# Let's extract unique features and select related to 'Basement' values

# In[ ]:


unique_basement_features = get_unique_values('Basement')
print(len(unique_basement_features), unique_basement_features)


# #### Finished (These basements or spaces are completed or usable, often with features like exterior access or fully finished rooms.)
# - "Full Basement": A basement that spans the entire footprint of the house and is fully developed for use.
# - "Full": Another way to refer to a completely finished basement.
# - "Dugout": A basement or below-ground space that has been excavated (dug out) to provide additional space.
# - "Fully Finished": A basement that is 100% completed with flooring, walls, ceiling, and sometimes utilities.
# - "Finished": A general term for a basement that is move-in ready.
# - "Remodeled Basement": A basement that was previously unfinished or outdated but has been renovated.
# - "Apartment in Basement": A basement that has been converted into a separate living unit with amenities like a kitchen and bathroom.
# - "Suite": A finished basement space designed as a separate living area, often with its own entrance.
# - "Walk-Out Access": A basement with a door leading directly outside, typically to ground level.
# - "Walk-Out To Grade": Similar to "Walk-Out Access," meaning the basement exits directly to ground level.
# - "Walk-Up To Grade": A basement that has a stairway leading up to ground level for external access.
# - "Walk-Out / Walkout": Another way to describe a basement that has an external door leading outside.
# - "Walk-Up": A basement with stairs leading to an outside entrance.
# - "With Windows": A basement that has windows, often allowing natural light and making it more livable.
# - "Separate/Exterior Entry": A basement that has a private entrance from outside, making it ideal for rental units or guest spaces.
# - "Separate Entrance": Similar to "Separate/Exterior Entry," meaning the basement has independent access from outside.
# 
# #### Partial (These basements or spaces are partially finished or incomplete.)
# - "Partially Finished" / "Partially finished": Some areas are finished, but others may be unfinished or under construction.
# - "Partial" / "Partial Basement": The basement is smaller than the home's full footprint or has limited finished space.
# - "Not Full Height": A basement that does not have standard ceiling height, making it less functional as a full living space.
# - "Cellar": An older term, often referring to basements used for storage, with lower ceilings and minimal finishing.
# 
# #### No Basement (These properties either don't have basements or have limited, unusable spaces underneath.)
# - "Unfinished": A basement that exists but has no completed flooring, walls, or ceiling.
# - "No Basement": The property does not include a basement.
# - "Slab": The house is built directly on a concrete slab, with no basement underneath.
# - "Crawl Space" / "Crawl": A shallow space under the house, not intended for living but used for ventilation or storage.
# - "None": No basement or similar structure exists in the property.
# - "N/A": Not available or not provided.
# - "Not Applicable": The basement condition does not apply, possibly because there is no basement or it's an irrelevant listing.

# In[ ]:


basement_types = {'Finished': ['full basement', 'full', 'dugout', 'fully finished', 'finished', 'remodeled basement', 
                               'apartment in basement', 'suite', 'walk-out access', 'walk-out to grade', 'walk-up to grade', 
                               'walk-out', 'walkout', 'walk-up', 'with windows', 'separate/exterior entry', 'separate entrance'], 
                  'Partial': ['partially finished', 'partial', 'partially finished', 'partial basement', 
                              'not full height', 'cellar'], 
                  'No basement': ['unfinished', 'no basement', 'slab', 'crawl space', 'crawl', 'none', 
                                  'not applicable', 'n/a']}


# In[ ]:


def check_type(features, types):

    try:
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "").replace(";", ",")
        list_to_check = [f.strip().lower() for f in cleaned_features.split(',')]

        for basement_type, basement_type_list in types.items():
            if set(list_to_check) & set(basement_type_list): 
                return basement_type
        return np.nan

    except AttributeError:
        return np.nan


# In[ ]:


df['Basement new'] = df['Basement']
df['Basement new'] = df['Basement new'].apply(lambda x: check_type(x, basement_types))
df['Basement new'].value_counts()


# ## Exterior

# In[ ]:


df.Exterior.value_counts().head()


# In[ ]:


unique_exterior_features = get_unique_values('Exterior')   
print(len(unique_exterior_features), unique_exterior_features)


# In[ ]:


exterior_materials = {
    'Metal': ['aluminum', 'aluminum siding', 'aluminum/vinyl', 'colour loc', 'metal', 'steel'],
    'Brick': ['brick', 'brick facing', 'brick imitation', 'brick veneer'],
    'Concrete': ['concrete', 'concrete siding', 'concrete/stucco', 'concrete block', 'insul brick', 'stucco'],
    'Wood': ['cedar', 'cedar shingles', 'cedar siding', 'wood', 'wood siding', 'wood shingles', 'wood shingles', 'wood siding'],
    'Composite': ['composite siding', 'hardboard', 'masonite', 'shingles'],
    'Vinyl': ['vinyl', 'vinyl siding', 'asbestos', 'siding']
}


# In[ ]:


df['Exterior new'] = df['Exterior']
df['Exterior new'] = df['Exterior new'].apply(lambda x: check_type(x, exterior_materials))
df['Exterior new'].value_counts()


# ## Fireplace

# In[ ]:


print(df.Fireplace.shape[0]) 
df.Fireplace.value_counts()[:10]


# Column 'Fireplace' contain information in different appearence. Let's reduce it ot 'Yes' and 'No'. Also we need to get information from the column 'Fireplace Features'.

# In[ ]:


df['Fireplace new'] = df['Fireplace']

df.loc[df['Fireplace new'].isin(['0', '[]','["0"]']), 'Fireplace new'] = np.nan
df.loc[df['Fireplace new'].notna(), 'Fireplace new'] = 'Yes'
df.loc[~df['Fireplace new'].notna(), 'Fireplace new'] = 'No'


# In[ ]:


df['Fireplace new'].value_counts()


# # Heating

# In[ ]:


df.Heating.unique().shape


# In[ ]:


unique_heating_features = get_unique_values('Heating')
print(len(unique_heating_features), unique_heating_features)


# Let's divide them in categories:

# In[ ]:


heating_categories = {
    'forced air': ['forced air', 'forced air-1', 'forced air-2', 'furnace'],
    'boiler': ['boiler', 'hot water', 'hot water radiator heat', 'steam', 'steam radiator'],
    'heat pump': ['central heat pump', 'heat pump', 'wall mounted heat pump'],
    'radiant': ['radiant', 'radiant heat', 'radiant ceiling', 'radiant floor', 
                'radiant/infra-red heat', 'baseboard', 'baseboard heaters', 'electric baseboard units'],
    'fireplace': ['fireplace(s)', 'fireplace insert', 'wood stove', 'pellet stove', 'coal stove', 'stove'],
    'space heat': ['space heater', 'space heaters', 'wall furnace', 'floor furnace', 
                   'floor model furnace', 'overhead heaters', 'overhead unit heater', 'ductless'],
    'alt heat': ['geo thermal', 'geothermal', 'solar', 'gravity', 'gravity heat system', 
                 'oil', 'propane', 'propane gas', 'coal'],
    'no heat': ['no heat', 'none'],  # only contains no heat options
    'other': ['mixed', 'combination', 'sep. hvac units'],  # these suggest non-specific or mixed systems
}


# In[ ]:


df['Heating new'] = df['Heating']
df['Heating new'] = df['Heating new'].apply(lambda x: check_type(x, heating_categories))

df['Heating new'] .value_counts()


# # Flooring

# In[ ]:


df.Flooring.value_counts()


# In[ ]:


unique_flooring_features = get_unique_values('Flooring')
print(len(unique_flooring_features), unique_flooring_features)


# Let's divide into next categories:
# - Carpet: Includes all variations of carpet.
# - Wood: Includes bamboo, engineered wood, hardwood, etc.
# - Tile: Covers ceramic, porcelain, slate, and other types of tiles.
# - Vinyl: Includes all vinyl-based flooring types.
# - Laminate: Groups laminate and laminate flooring.
# - Concrete: Includes concrete and concrete slabs.
# - Other: For flooring types that are not easily categorized (like marble, granite, subfloor, etc.).

# In[ ]:


flooring_categories = {
    'carpet': ['carpet', 'carpet over softwood', 'carpet over hardwood', 'carpeted', 'wall to wall carpet', 
               'wall-to-wall carpet'],
    'wood': ['bamboo', 'engineered wood', 'engineered hardwood', 'hardwood', 'parquet', 'softwood', 'wood'],
    'tile': ['ceramic', 'ceramic tile', 'ceramic/porcelain', 'porcelain tile', 'non-ceramic tile', 'slate', 
             'stone', 'tile'],
    'vinyl': ['cushion/lino/vinyl', 'vinyl', 'vinyl plank'],
    'laminate': ['laminate', 'laminate flooring'],
    'concrete': ['concrete', 'concrete slab'],
    'other': ['basement slab', 'basement sub-floor', 'granite', 'heavy loading', 'linoleum', 'marble', 'mixed', 
              'mixed flooring', 'see remarks', 'subfloor', 'other']
}


# In[ ]:


df['Flooring new'] = df['Flooring']
df['Flooring new'] = df['Flooring new'].apply(lambda x: check_type(x, flooring_categories))

df['Flooring new'].value_counts()


# # Roof

# In[ ]:


df.Roof.value_counts()


# In[ ]:


unique_roof_features = get_unique_values('Roof')
print(len(unique_roof_features), unique_roof_features)


# Categories:
# - Asphalt: Includes all types of asphalt and asphalt shingles.
# - Cedar Shake: Includes cedar shake and shakes.
# - Clay: For clay tiles.
# - Concrete: Covers concrete and concrete tiles.
# - Fiberglass: Includes fiberglass and fiberglass shingles.
# - Flat: Includes flat roofing, EPDM, membrane, and flat torch membrane types.
# - Metal: Covers metal, metal shingles, steel, and tin.
# - Pine Shake: Includes pine shake and shakes.
# - Rubber: For rubber roofing.
# - SBS: For SBS roofing system.
# - Shake: Covers shake roofing.
# - Shingle: Includes regular shingles and vinyl shingles.
# - Slate: For slate roofing.
# - Tar: Includes various types of tar and gravel roofing.
# - Tile: For tile roofing.
# - Wood: Covers wood shingles and wood shake.
# - Other: For unconventional or mixed roofing, or cases where the material is unclear (like 'conventional', 'other', 'mixed').

# In[ ]:


roofing_categories = {
    'asphalt': ['asphalt', 'asphalt & gravel', 'asphalt rolled', 'asphalt shingle', 'asphalt shingles', 'asphalt torch on', 'asphalt shingle', 'asphalt/gravel'],
    'cedar shake': ['cedar shake', 'cedar shakes'],
    'clay': ['clay tile'],
    'concrete': ['concrete', 'concrete tiles'],
    'fiberglass': ['fiberglass', 'fiberglass shingles', 'fibreglass shingle'],
    'flat': ['flat', 'flat torch membrane', 'membrane', 'epdm membrane'],
    'metal': ['metal', 'metal shingles', 'steel', 'tin'],
    'pine shake': ['pine shake', 'pine shakes'],
    'rubber': ['rubber'],
    'sbs': ['sbs roofing system'],
    'shake': ['shake'],
    'shingle': ['shingle', 'vinyl shingles'],
    'slate': ['slate'],
    'tar': ['tar & gravel', 'tar & gravel', 'tar &amp; gravel', 'tar/gravel'],
    'tile': ['tile'],
    'wood': ['wood', 'wood shingle', 'wood shingles'],
    'other': ['conventional', 'mixed', 'other'],
}


# In[ ]:


df['Roof new'] = df['Roof']
df['Roof new'] = df['Roof new'].apply(lambda x: check_type(x, roofing_categories))
df['Roof new'].value_counts()


# # Property Type

# In[ ]:


df['Property Type'].value_counts()


# # Waterfront

# In[ ]:


df.Waterfront.value_counts()


# In[ ]:


df['Waterfront new'] = df['Waterfront']
df.loc[df['Waterfront new'].isna(), 'Waterfront new'] = 'No'

df['Waterfront new'].value_counts()


# # Subdivision

# In[ ]:


df.Subdivision.value_counts()[:30]


# # Sewer

# In[ ]:


df.Sewer.value_counts()


# In[ ]:


unique_sewer_features = get_unique_values('Sewer')
print(len(unique_sewer_features), unique_sewer_features)


# In[ ]:


sewage_categories = {
    'municipal': ['municipal/community', 'municipal sewage system', 'sanitary sewer', 'sewer', 'sewer connected', 
                  'sewer to lot', 'sewer available', 'public sewer', 'attached to municipal'],
    'septic': ['septic tank', 'septic system', 'septic system: common', 'septic field', 'septic tank and field', 
               'septic tank & mound', 'mound septic', 'septic tank & field', 'septic needed', 'engineered septic'],
    'private': ['private sewer', 'private sewer', 'holding tank', 'low pressure sewage sys', 'shared septic'],
    'alternative': ['aerobic treatment system', 'facultative lagoon', 'lagoon', 'outflow tank', 'open discharge', 
                    'liquid surface dis', 'pump', 'tank & straight discharge'],
    'none': ['no sewage system', 'outhouse', 'none'],
}


# In[ ]:


df['Sewer new'] = df['Sewer']
df['Sewer new'] = df['Sewer new'].apply(lambda x: check_type(x, sewage_categories))
df['Sewer new'] = df['Sewer new'].fillna('none')

df['Sewer new'].value_counts()


# # Additional Features 

# Let's add some additional features like Pool, Gerden, View and Balcony. 

# In[ ]:


mix_features = ['Basement', 'Exterior', 'Features', 'Fireplace', 'Garage', 'Heating', 'Parking']


# In[ ]:


df['Combined'] = df[mix_features].astype(str).agg(','.join, axis=1)
df.head(1)


# In[ ]:


##### Extract unique features
unique_mix_features = set()

for features in df['Combined'].dropna():
    # Remove brackets and extra quotes
    cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "")
    # Split by comma
    feature_list = [f.strip() for f in cleaned_features.split(",")]
    unique_mix_features.update(feature_list)

print(len(unique_mix_features), unique_mix_features)


# # Pool

# Let's add Pool feature

# In[ ]:


pool_features = ["swimming pool", "public swimming pool"]


# In[ ]:


df['Pool'] = df['Combined'].apply(lambda x: check_keywords(x, pool_features))
df['Pool'].value_counts()


# # Garden

# In[ ]:


garden_features = ["vegetable garden", "garden", "fruit trees/shrubs", "private yard", 
                   "partially landscaped", "landscaped"]


# In[ ]:


df['Garden'] = df['Combined'].apply(lambda x: check_keywords(x, garden_features))
df['Garden'].value_counts()


# # View

# In[ ]:


view_features = ["View Downtown", "River View", "View City", "View Lake", "Lake View", 
                 "Ravine View", "River Valley View"]

view_features = [f.lower() for f in view_features]


# In[ ]:


def check_view(features, keywords):

    try:
        cleaned_features = features.replace("[", "").replace("]", "").replace('"', '').replace("'", "").replace(";", ",")
        list_to_check = [f.strip().lower() for f in cleaned_features.split(',')]
        common = set(list_to_check) & set(keywords)

        if common: 
            return list(common)[0]  

    except AttributeError:
        return np.nan

    return np.nan


# In[ ]:


df['View'] = df['Combined'].apply(lambda x: check_view(x, view_features))
df['View'].value_counts()


# In[ ]:


df.loc[df['View'].isin(['view lake', 'lake view']), 'View'] = 'Lake'
df.loc[df['View'].isin(['view downtown']), 'View'] = 'Downtown'
df.loc[df['View'].isin(['view city']), 'View'] = 'City'
df.loc[df['View'].isin(['river view']), 'View'] = 'River'
df.loc[df['View'].isin(['ravine view', 'river valley view']), 'View'] = 'Valley'

df['View'].value_counts()


# # Balcony

# In[ ]:


balcony_features = ['Balcony', 'Balcony/Deck', 'Balcony/Patio']
balcony_features = [f.lower() for f in balcony_features]


# In[ ]:


df['Balcony'] = df['Combined'].apply(lambda x: check_keywords(x, balcony_features))
df['Balcony'].value_counts()


# In[ ]:


df = df.drop(columns=['streetAddress', 'postalCode', 'description', 'priceCurrency', 'Air Conditioning', 
                 'Basement', 'Exterior', 'Features', 'Fireplace', 'Garage', 'Heating', 'MLS® #', 'Roof', 
                 'Sewer', 'Waterfront', 'Parking', 'Flooring', 'Fireplace Features', 'Combined', 
                 'Subdivision', 'property-sqft', 'Square Footage', 'Bath', 'Property Tax'])


# In[ ]:


df = df.rename(columns={'addressLocality': 'City', 
                        'addressRegion': 'Province', 
                        'latitude': 'Latitude', 
                        'longitude': 'Longitude', 
                        'price': 'Price',
                        'property-baths': 'Bathrooms', 
                        'property-beds': 'Bedrooms', 
                        'Square Footage new': 'Square Footage', 
                        'Garage new': 'Garage',	
                        'Parking new': 'Parking',	
                        'Basement new': 'Basement', 
                        'Exterior new': 'Exterior', 
                        'Fireplace new': 'Fireplace', 
                        'Heating new': 'Heating', 
                        'Flooring new': 'Flooring', 
                        'Roof new': 'Roof', 
                        'Waterfront new': 'Waterfront', 
                        'Sewer new': 'Sewer'})


# In[ ]:


df.info()


# # Types

# In[ ]:


df['Square Footage'] = df['Square Footage'].str.replace(',','')


# In[ ]:


number_columns = ['Latitude', 'Longitude', 'Price', 'Bedrooms', 'Bathrooms', 'Acreage', 
                  'Square Footage']

for col in number_columns: 
    df[col] = df[col].astype(float)


# In[ ]:


df.info()


# # Remove inconsistent data

# In[ ]:


data = df.copy()


# ### Square Footage, Bedrooms, and Bathrooms

# Let's remove all houses with out Bedroom or Bathrooms

# In[ ]:


data.dropna(subset=["Bedrooms", "Bathrooms", "Square Footage"], inplace=True)
data[["Bedrooms", "Bathrooms", "Square Footage"]].describe()


# Remove also rows with 'Square Footage' less then 120 sqft.

# In[ ]:


data = data[data["Square Footage"] > 120]


# Remove rows with no bedrooms or bathrooms

# In[ ]:


# data = data[(data["Bedrooms"] > 0) & (data["Bathrooms"] > 0)]


# In[ ]:


data[["Bedrooms", "Bathrooms", "Square Footage"]].describe()


# ### Province and City

# In[ ]:


data.Province.value_counts()


# In[ ]:


data.City.unique().shape


# In[ ]:


data.City.value_counts()[:10]


# We have 3112 different cities in dataset. Perhebs, we will not use this column while modeling

# ### Price

# In[ ]:


data.Price.value_counts()


# We have a lot of houses with unrealistic low price. Perhebs. some position from rent accidantly were put inro sell patr. Let's delete all houses with price less then 50 000.

# In[ ]:


data = data[data.Price >= 50_000]


# ### Property Type

# In[ ]:


data['Property Type'].value_counts()


# ### Garage, Parking, 

# In[ ]:


data[['Garage', 'Parking']].info()


# In[ ]:


data['Garage'].value_counts()


# In[ ]:


data['Parking'].value_counts()


# ### Fireplace, Pool, Garden, Balcony

# In[ ]:


data[['Fireplace', 'Pool', 'Garden', 'Balcony', 'Sewer']].info()


# ### Basement, Exterior, Fireplace, Heating, Flooring, Roof, Sewer, View

# In[ ]:


data[['Basement', 'Exterior', 'Heating', 'Flooring', 'Roof', 'View']].info()


# In[ ]:


data = data.drop(columns=['View'])
data.head()


# In[ ]:





# # Save data

# In[ ]:


df.to_csv('cleaned_canada.csv', index=False)


# In[ ]:




