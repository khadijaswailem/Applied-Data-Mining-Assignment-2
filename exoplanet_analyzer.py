import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()#loading environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")#getting GROQ API key

#File Path
file_path = r"C:\Users\KhadijaSwailem\Downloads\applieddatamining\assignment 2\PS_2026.03.02_08.26.20.csv"

#class definition
class ExoplanetAnalyzer:
    #constructor to initialize GROQ client
    def __init__(self, file_path):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.data = self._load_exoplanet_data(file_path)

    #loading Exoplanet dataset
    def _load_exoplanet_data(self, file_path):
        df = pd.read_csv(file_path, comment='#')

        #only relevant columns to consider token limit 
        selected_columns = [
            'pl_name',
            'hostname',
            'discoverymethod',
            'disc_year',
            'pl_orbper',
            'pl_rade',
            'pl_masse',
            'pl_eqt'
        ]

        df = df[selected_columns]
        return df

    #analyzing single planet 
    def analyze_planet(self, planet_name):
        #AI based analyisis of the planet
        #querying dataset for specific planet
        planet_data = self.data[self.data['pl_name'] == planet_name]

        #error handling
        if planet_data.empty:#if planet doesnt have data in the dataset
            return f"Planet '{planet_name}' not found in dataset."

        planet_data = planet_data.iloc[0]#getting planet row as series for prompt


        #missing value handling
        missing_cols = planet_data[['pl_orbper','pl_rade','pl_masse','pl_eqt']].isnull()
        if missing_cols.any():
            missing_list = list(missing_cols[missing_cols].index)
            warning_msg = f"Note: Missing data for {', '.join(missing_list)}."
        else:
            warning_msg = ""

        #creating the prompt using func we defined below
        prompt = self._create_analysis_prompt(planet_data)

        #if there are missing values, we add a warning to the prompt to inform the AI about the incomplete data, which can help it generate a better analysis based on the available information
        if warning_msg:
            prompt = warning_msg + "\n\n" + prompt
        try:
            response = self.client.chat.completions.create(#the ananlysis
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an astrophysicist providing scientific planetary analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            # Return AI response text
            return response.choices[0].message.content
        except Exception as e:
            return f"Error while analyzing planet: {e}"
        
       

    #prompt template for each planet 
    def _create_analysis_prompt(self, planet_data):

        #creating clear and concise scientific prompt for planet analysis

        prompt = f"""
        Provide a structured scientific analysis of the following exoplanet.

        Planet Name: {planet_data['pl_name']}
        Host Star: {planet_data['hostname']}
        Discovery Method: {planet_data['discoverymethod']}
        Discovery Year: {planet_data['disc_year']}
        Orbital Period (days): {planet_data['pl_orbper']}
        Radius (Earth radii): {planet_data['pl_rade']}
        Mass (Earth masses): {planet_data['pl_masse']}
        Equilibrium Temperature (K): {planet_data['pl_eqt']}

        Structure the response using these sections:
        1. Basic Characteristics
        2. Orbital Properties
        3. Physical Composition Implications
        4. Habitability Considerations
        5. Scientific Significance

        Use formal, scientific language.
        Avoid speculation beyond provided data.
        """

        return prompt

    #comparing multiple planets
    def compare_planets(self, planet_names):
        #generating comparison analysis between multiple planets

        planets = self.data[self.data['pl_name'].isin(planet_names)]#filtering dataset for specific planets

        if planets.empty:#if none of the specific planets are found in the dataset
            return "None of the specified planets were found."

        
        prompt = self._create_comparison_prompt(planets)#again calling func to create structured prompt for comparison

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                {"role": "system", "content": "You are an astrophysicist comparing exoplanets scientifically."},
                {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            #returns first response
            return response.choices[0].message.content
        except Exception as e:
            return f"Error while comparing planets: {e}"
        

    #prompt template for planet comparison
    def _create_comparison_prompt(self, planets):
        #creating structured comparison prompt

        planet_info = ""#initializing empty string for planet details for the prompt

        for _, planet in planets.iterrows():#iterating through each planet and appending its details to the prompt 
            planet_info += f"""
            Planet Name: {planet['pl_name']}
            Host Star: {planet['hostname']}
            Discovery Method: {planet['discoverymethod']}
            Discovery Year: {planet['disc_year']}
            Orbital Period (days): {planet['pl_orbper']}
            Radius (Earth radii): {planet['pl_rade']}
            Mass (Earth masses): {planet['pl_masse']}
            Equilibrium Temperature (K): {planet['pl_eqt']}
            -----------------------------
            """

        #the prompt to guide the AI in making a scientific comparison between the planets based on the provided data, with specific sections to ensure a comprehensive analysis.
        prompt = f"""
        Compare the following exoplanets scientifically.

        {planet_info}

        Structure the comparison under these headings:
        1. Size and Mass Comparison
        2. Orbital Characteristics
        3. Temperature Differences
        4. Potential Composition Differences
        5. Habitability Comparison
        6. Key Scientific Insights

        Maintain a formal scientific tone.
        Focus only on provided numerical data.
        """

        return prompt