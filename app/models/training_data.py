"""
Pequeno dataset en espanol para entrenar el clasificador de incidentes.
Lo armamos nosotros con frases tipicas de los grupos de WhatsApp de barrio.
Las etiquetas son los mismos codigos de la tabla incident_types.
"""

TRAINING_DATA = [
    # armed_robbery
    ("me atracaron con un cuchillo y se llevaron el celular", "armed_robbery"),
    ("dos tipos en moto armados con pistola robaron a una pareja", "armed_robbery"),
    ("asalto a mano armada en la tienda de la esquina", "armed_robbery"),
    ("le pusieron un revolver en la cabeza al conductor y le quitaron todo", "armed_robbery"),
    ("atraco con arma de fuego a un domiciliario le robaron la moto", "armed_robbery"),
    ("nos amenazaron con navaja para quitarnos las pertenencias", "armed_robbery"),
    ("robo con arma blanca en el puente peatonal", "armed_robbery"),
    ("hombres armados asaltaron el bus y dispararon al aire", "armed_robbery"),
    ("atracaron al vigilante con una pistola", "armed_robbery"),
    ("fleteo a la salida del banco con arma de fuego", "armed_robbery"),
    # assault
    ("rina en la salida del bar hay un herido", "assault"),
    ("varias personas se agarraron a golpes en la calle", "assault"),
    ("un hombre golpeo a una mujer en el parque", "assault"),
    ("pelea entre barras con piedras y palos", "assault"),
    ("agredieron a un vecino y lo dejaron sangrando", "assault"),
    ("le pegaron a un adulto mayor para intimidarlo", "assault"),
    ("hubo una pelea con botellas frente a la discoteca", "assault"),
    ("agresion fisica a un conductor por un choque", "assault"),
    # home_burglary
    ("se metieron a una casa y se llevaron el televisor", "home_burglary"),
    ("forzaron la puerta del apartamento mientras no habia nadie", "home_burglary"),
    ("entraron por la ventana y robaron el computador", "home_burglary"),
    ("robaron una vivienda rompiendo la chapa de la reja", "home_burglary"),
    ("ladrones saltaron el muro y saquearon la casa", "home_burglary"),
    ("robo en apartamento del primer piso de madrugada", "home_burglary"),
    ("se entraron al local comercial por el techo", "home_burglary"),
    ("vaciaron la casa del vecino mientras estaba de viaje", "home_burglary"),
    # vehicle_theft
    ("se robaron un carro parqueado en la calle", "vehicle_theft"),
    ("hurtaron una moto frente a la panaderia", "vehicle_theft"),
    ("rompieron el vidrio del carro y se lo llevaron", "vehicle_theft"),
    ("robo de bicicleta en el parqueadero del conjunto", "vehicle_theft"),
    ("se llevaron las llantas y los espejos del carro", "vehicle_theft"),
    ("le robaron la camioneta al senor de la esquina", "vehicle_theft"),
    ("desvalijaron un vehiculo estacionado", "vehicle_theft"),
    ("robaron la motocicleta del domiciliario sin armas", "vehicle_theft"),
    # theft
    ("raponazo de celular en el paradero del bus", "theft"),
    ("le sacaron la billetera del bolsillo en transmilenio", "theft"),
    ("cosquilleo en el centro comercial se llevaron un bolso", "theft"),
    ("le arrebataron la cadena a una senora", "theft"),
    ("hurto de celular a un estudiante saliendo del colegio", "theft"),
    ("robaron el bolso de una mujer sin que se diera cuenta", "theft"),
    ("le quitaron el celular corriendo y se perdio entre la gente", "theft"),
    ("hurto de pertenencias en el mercado", "theft"),
    ("le robaron el morral a un nino en el parque", "theft"),
    # drug_dealing
    ("venta de droga en el parque todas las noches", "drug_dealing"),
    ("olla de vicio en la casa de la esquina", "drug_dealing"),
    ("jibaros vendiendo sustancias cerca al colegio", "drug_dealing"),
    ("mucho movimiento de gente comprando drogas", "drug_dealing"),
    ("expendio de marihuana en la cancha", "drug_dealing"),
    ("estan consumiendo y vendiendo bazuco debajo del puente", "drug_dealing"),
    ("microtrafico en la entrada del conjunto", "drug_dealing"),
    # vandalism
    ("rompieron las lamparas del parque y quedo oscuro", "vandalism"),
    ("rayaron las paredes del colegio con grafitis", "vandalism"),
    ("rompieron vidrios de los carros parqueados", "vandalism"),
    ("danaron los juegos infantiles del parque", "vandalism"),
    ("quemaron una caneca de basura en la calle", "vandalism"),
    ("destruyeron el paradero del bus", "vandalism"),
    ("pintaron los muros y rompieron las bancas", "vandalism"),
    # suspicious_activity
    ("persona sospechosa mirando las casas y tomando fotos", "suspicious_activity"),
    ("carro sin placas parqueado con dos personas adentro", "suspicious_activity"),
    ("un hombre lleva horas rondando el conjunto", "suspicious_activity"),
    ("moto con dos personas dando vueltas por la cuadra", "suspicious_activity"),
    ("sujetos extranos preguntando por los vecinos", "suspicious_activity"),
    ("alguien intento abrir la puerta y salio corriendo", "suspicious_activity"),
    ("personas sospechosas vigilando la entrada del edificio", "suspicious_activity"),
    ("un tipo merodeando los carros del parqueadero", "suspicious_activity"),
]
