import json
import os
import sys
from pathlib import Path

def merge_world_bank_into_sipri(sipri_file_path, wb_file_path, output_path):
    """
    Merge World Bank indicators data into SIPRI military expenditure data.
    
    This script:

    - Loads both the SIPRI military expenditure data and the World Bank data
    - Adds World Bank indicators directly to each country's time_series data
    - Uses indicator descriptions instead of codes for better readability
    - Removes the arms_trading nested structure while keeping total counts
    - Preserves the structure of the SIPRI data
    - Handles missing data and provides detailed logging
    - Creates an indicators key at the top level of the output JSON
    - Saves the merged data to a new JSON file
    - Ensures each country has ALL indicators (with NULL for missing values)
    
    Args:
        sipri_file_path: Path to the SIPRI military expenditure JSON file
        wb_file_path: Path to the World Bank indicators JSON file
        output_path: Path where the merged JSON file will be saved
    """
    try:
        # Load the SIPRI military expenditure data
        with open(sipri_file_path, 'r') as f:
            sipri_data = json.load(f)
        
        # Load the World Bank data
        with open(wb_file_path, 'r') as f:
            wb_data = json.load(f)
        
        # Add World Bank metadata to the output
        sipri_data['world_bank_metadata'] = wb_data.get('metadata', {})
        
        # Create indicator code to name mapping
        indicators_map = wb_data.get('indicators', {})
        
        # Add indicators key at the top level of the output JSON
        sipri_data['indicators'] = {
            "SIPRI": {},
            "World Bank": indicators_map
        }
        
        # Create a dictionary for faster lookup of World Bank data by ISO code
        wb_data_by_iso = {}
        for country in wb_data['countries']:
            wb_data_by_iso[country['ISO']] = country['data']
        
        # Track statistics for reporting
        total_countries = len(sipri_data['countries'])
        countries_with_wb_data = 0
        indicators_added = 0
        arms_trading_removed = 0
        sipri_indicators = set()
        
        # First pass: Collect all possible indicators from both datasets
        all_indicators = set()
        
        # Collect SIPRI indicators first
        for country in sipri_data['countries']:
            for time_series_entry in country.get('time_series', []):
                for key in time_series_entry.keys():
                    if key not in ['year', 'date', 'iso', 'country', 'arms_trading']:
                        all_indicators.add(key)
        
        # Collect World Bank indicator names
        wb_indicator_names = set()
        for code, info in indicators_map.items():
            indicator_name = info.get('name', code)
            wb_indicator_names.add(indicator_name)
            all_indicators.add(indicator_name)
        
        # Add World Bank data to each country in the SIPRI data
        for country in sipri_data['countries']:
            iso_code = country['ISO']
            
            # Track if this country has any World Bank data
            has_wb_data = False
            
            # Process each time series entry to add World Bank indicators
            for time_series_entry in country.get('time_series', []):
                # Remove the arms_trading nested structure
                if 'arms_trading' in time_series_entry:
                    del time_series_entry['arms_trading']
                    arms_trading_removed += 1
                
                year = time_series_entry.get('year')
                
                # Skip if no year is found
                if year is None:
                    # If no explicit year, try to find a year field in the entry
                    for key in time_series_entry:
                        if key.lower().endswith('year'):
                            year = time_series_entry[key]
                            break
                
                # Add World Bank data if available for this country and year
                if iso_code in wb_data_by_iso and year:
                    country_wb_data = wb_data_by_iso[iso_code]
                    has_wb_data = True
                    
                    # Process each indicator in the World Bank data
                    for indicator_code, indicator_data in country_wb_data.items():
                        year_str = str(year)
                        if year_str in indicator_data:
                            # Get indicator name (description) instead of code
                            if indicator_code in indicators_map:
                                indicator_name = indicators_map[indicator_code].get('name', indicator_code)
                                time_series_entry[indicator_name] = indicator_data[year_str]
                                indicators_added += 1
                
                # Collect SIPRI indicators
                for key in time_series_entry:
                    if key not in ['year', 'date', 'iso', 'country']:
                        # Check if this is likely a SIPRI indicator
                        sipri_keywords = ['milex', 'military', 'expenditure', 'arms', 'weapon', 'defense', 'defence']
                        if any(keyword in key.lower() for keyword in sipri_keywords):
                            sipri_indicators.add(key)
                            
                # Ensure all indicators exist (add NULL for missing ones)
                for indicator in all_indicators:
                    if indicator not in time_series_entry:
                        time_series_entry[indicator] = None
            
            # Count country with WB data once per country, not once per year
            if has_wb_data:
                countries_with_wb_data += 1
            else:
                print(f"Warning: No World Bank data found for {country['name']} (ISO: {iso_code})")
        
        # Add SIPRI indicators to the indicators key
        for indicator in sipri_indicators:
            sipri_data['indicators']["SIPRI"][indicator] = {
                "name": indicator,
                "description": ""
            }
        
        # Check for countries in World Bank data not in SIPRI data
        sipri_iso_codes = {country['ISO'] for country in sipri_data['countries']}
        wb_iso_codes = set(wb_data_by_iso.keys())
        missing_countries = wb_iso_codes - sipri_iso_codes
        if missing_countries:
            print(f"Note: {len(missing_countries)} countries in World Bank data do not exist in SIPRI data")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the merged data to a new JSON file
        with open(output_path, 'w') as f:
            json.dump(sipri_data, f, indent=2)
        
        print(f"Data merge completed successfully.")
        print(f"Total countries in SIPRI data: {total_countries}")
        print(f"Countries with World Bank data: {countries_with_wb_data}")
        print(f"Total World Bank indicators added: {indicators_added}")
        print(f"Total indicators in dataset: {len(all_indicators)}")
        print(f"SIPRI indicators identified: {len(sipri_indicators)}")
        print(f"Arms trading structures removed: {arms_trading_removed}")
        print(f"Output saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def extract_indicators_key(merged_data_path, output_key_path):
    """
    Extract all available indicators from the merged data and create a JSON key file.
    
    This function:
    - Loads the merged SIPRI and World Bank data
    - Extracts all unique indicator names
    - Provides descriptions and categorizes indicators by source
    - Saves the indicator key to a JSON file
    
    Args:
        merged_data_path: Path to the merged data JSON file
        output_key_path: Path where the indicator key JSON file will be saved
    """
    try:
        # Load the merged data
        with open(merged_data_path, 'r') as f:
            merged_data = json.load(f)
        
        # Initialize sets to track unique indicators
        all_indicators = set()
        
        # Get World Bank indicator metadata (codes and descriptions)
        wb_indicator_metadata = {}
        if 'world_bank_metadata' in merged_data and 'indicators' in merged_data['world_bank_metadata']:
            wb_indicator_metadata = merged_data['world_bank_metadata']['indicators']
        
        # Extract all unique indicators from the data
        for country in merged_data.get('countries', []):
            for time_entry in country.get('time_series', []):
                for key in time_entry.keys():
                    # Skip standard fields like 'year'
                    if key not in ['year', 'date', 'iso', 'country']:
                        all_indicators.add(key)
        
        # Create structured indicator dictionary
        indicators_dict = {}
        
        # Define SIPRI-related keywords to identify SIPRI indicators
        sipri_keywords = ['milex', 'military', 'expenditure', 'arms', 'weapon', 'defense', 'defence']
        
        # Categorize each indicator
        for indicator in all_indicators:
            # Determine if it's a SIPRI or World Bank indicator
            source = "SIPRI" if any(keyword in indicator.lower() for keyword in sipri_keywords) else "World Bank"
            
            # Find the indicator code if it's a World Bank indicator
            indicator_code = "unknown"
            description = ""
            for code, info in wb_indicator_metadata.items():
                if info.get('name') == indicator:
                    indicator_code = code
                    description = info.get('description', '')
                    break
            
            # Add to dictionary
            indicators_dict[indicator] = {
                "source": source,
                "code": indicator_code if source == "World Bank" else None,
                "description": description
            }
        
        # Group by source
        grouped_indicators = {
            "SIPRI": {},
            "World Bank": {}
        }
        
        for name, info in indicators_dict.items():
            grouped_indicators[info["source"]][name] = {
                "code": info["code"],
                "description": info["description"]
            }
        
        # Create the final indicator key structure
        indicators_key = {
            "total_indicators": len(indicators_dict),
            "sources": {
                "SIPRI": {
                    "count": len(grouped_indicators["SIPRI"]),
                    "indicators": grouped_indicators["SIPRI"]
                },
                "World Bank": {
                    "count": len(grouped_indicators["World Bank"]),
                    "indicators": grouped_indicators["World Bank"]
                }
            }
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_key_path), exist_ok=True)
        
        # Save the indicator key to a JSON file
        with open(output_key_path, 'w') as f:
            json.dump(indicators_key, f, indent=2)
        
        print(f"Indicator key extraction completed successfully.")
        print(f"Total unique indicators: {indicators_key['total_indicators']}")
        print(f"SIPRI indicators: {len(grouped_indicators['SIPRI'])}")
        print(f"World Bank indicators: {len(grouped_indicators['World Bank'])}")
        print(f"Output saved to: {output_key_path}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    # Define file paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent  # Assuming script is in backend/ directory
    
    sipri_file_path = repo_root / "data" / "sipri_milex_data_nested.json"
    wb_file_path = repo_root / "data" / "world_bank" / "world_bank_data_normalized.json"
    output_path = repo_root / "data" / "all_data_merged.json"
    
    merge_world_bank_into_sipri(sipri_file_path, wb_file_path, output_path)