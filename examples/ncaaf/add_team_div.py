import pandas as pd

def add_division_and_conference(df: pd.DataFrame, team_col_name: str, prefix:str = '') -> pd.DataFrame:
    """
    Adds `division` (FBS/FCS) and `conference` columns based on team abbreviation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    team_col_name : str
        Column name containing team abbreviations

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns
    """

    # --- FBS CONFERENCES ---
    fbs_conferences = {
        # Big Ten
        "ILL": "Big Ten", "WIS": "Big Ten", "MICH": "Big Ten", "OSU": "Big Ten",
        "IOWA": "Big Ten", "MINN": "Big Ten", "PSU": "Big Ten", "NEB": "Big Ten",
        "PUR": "Big Ten", "MSU": "Big Ten", "RUTG": "Big Ten", "MD": "Big Ten",
        "NW": "Big Ten", "IND": "Big Ten", "UCLA": "Big Ten", "USC": "Big Ten",

        # SEC
        "ALA": "SEC", "UGA": "SEC", "LSU": "SEC", "AUB": "SEC", "TENN": "SEC",
        "FLA": "SEC", "ARK": "SEC", "MISS": "SEC", "MSST": "SEC", "MIZZ": "SEC",
        "SCAR": "SEC", "UK": "SEC", "TXAM": "SEC", "VAN": "SEC",

        # ACC
        "CLEM": "ACC", "FSU": "ACC", "UNC": "ACC", "DUKE": "ACC", "NCST": "ACC",
        "UVA": "ACC", "VT": "ACC", "GT": "ACC", "WAKE": "ACC", "SYR": "ACC",
        "BC": "ACC", "LOU": "ACC", "PITT": "ACC", "MIA": "ACC",

        # Big 12
        "TEX": "Big 12", "OKLA": "Big 12", "TCU": "Big 12", "BAY": "Big 12",
        "OKST": "Big 12", "KSU": "Big 12", "KU": "Big 12", "ISU": "Big 12",
        "WVU": "Big 12", "UCF": "Big 12", "CIN": "Big 12", "HOU": "Big 12",
        "BYU": "Big 12", "TTU": "Big 12",

        # Pac-12 / former Pac teams
        "ORE": "Pac-12", "WASH": "Pac-12", "CAL": "Pac-12", "STAN": "Pac-12",
        "ORST": "Pac-12", "WSU": "Pac-12", "ARIZ": "Pac-12", "ASU": "Pac-12",
        "UTAH": "Pac-12", "COLO": "Pac-12",

        # AAC
        "SMU": "AAC", "MEM": "AAC", "USF": "AAC", "ECU": "AAC",
        "FAU": "AAC", "UTSA": "AAC", "NAVY": "AAC", "TEM": "AAC",

        # Mountain West
        "BSU": "MWC", "SDSU": "MWC", "UNLV": "MWC", "NEV": "MWC",
        "WYO": "MWC", "USU": "MWC", "CSU": "MWC", "FRES": "MWC",
        "SJSU": "MWC", "UNM": "MWC",

        # Sun Belt
        "APP": "Sun Belt", "TROY": "Sun Belt", "ULL": "Sun Belt",
        "GASO": "Sun Belt", "GAST": "Sun Belt", "USA": "Sun Belt",
        "ODU": "Sun Belt", "JMU": "Sun Belt",

        # MAC
        "BGSU": "MAC", "NIU": "MAC", "EMU": "MAC", "WMU": "MAC",
        "CMU": "MAC", "KENT": "MAC", "AKR": "MAC", "BUFF": "MAC",
        "BALL": "MAC", "M-OH": "MAC", "OHIO": "MAC", "TOL": "MAC",

        # Independents
        "ND": "Independent", "ARMY": "Independent", "UCONN": "Independent"
    }

    # --- FCS CONFERENCES ---
    fcs_conferences = {
        # MVFC
        "NDSU": "MVFC", "SDSU": "MVFC", "UNI": "MVFC", "YSU": "MVFC",
        "SIU": "MVFC", "ILST": "MVFC", "INDS": "MVFC", "UND": "MVFC",

        # Big Sky
        "MONT": "Big Sky", "MTST": "Big Sky", "EWU": "Big Sky",
        "SAC": "Big Sky", "NAU": "Big Sky", "IDST": "Big Sky",

        # CAA
        "DEL": "CAA", "VILL": "CAA", "RICH": "CAA", "WM": "CAA",
        "UNH": "CAA", "URI": "CAA", "ELON": "CAA",

        # SWAC
        "GRAM": "SWAC", "FAMU": "SWAC", "PV": "SWAC", "AAMU": "SWAC",

        # Ivy
        "HARV": "Ivy", "YALE": "Ivy", "PRIN": "Ivy", "DART": "Ivy",
        "BRWN": "Ivy", "COLG": "Ivy", "PENN": "Ivy", "COR": "Ivy",

        # Patriot
        "LEH": "Patriot", "HC": "Patriot", "LAF": "Patriot"
    }

    def get_conference(team):
        if team in fbs_conferences:
            return fbs_conferences[team]
        if team in fcs_conferences:
            return fcs_conferences[team]
        return "Unknown"

    def get_division(team):
        if team in fbs_conferences:
            return "FBS"
        if team in fcs_conferences:
            return "FCS"
        return "Unknown"

    df = df.copy()
    df[f"{prefix}conference"] = df[team_col_name].map(get_conference)
    df[f"{prefix}division"] = df[team_col_name].map(get_division)

    return df
