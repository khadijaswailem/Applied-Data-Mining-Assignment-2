from exoplanet_analyzer import ExoplanetAnalyzer


file_path = r"C:\Users\KhadijaSwailem\Downloads\applieddatamining\assignment 2\PS_2026.03.02_08.26.20.csv"

if __name__ == "__main__":
    analyzer = ExoplanetAnalyzer(file_path)

    #valid planet
    print(analyzer.analyze_planet("Kepler-22 b"))

    #invalid planet
    print(analyzer.analyze_planet("Nonexistent Planet"))

    #comparison
    print(analyzer.compare_planets(["Kepler-22 b", "Kepler-442 b"]))

    #planet with missing data
    analyzer.data.loc[0, 'pl_rade'] = None
    print(analyzer.analyze_planet(analyzer.data.loc[0, 'pl_name']))