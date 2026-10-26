"""Catálogo peruano de entidades y escenarios, y detección de enlaces (sección 2.2).

Código tomado del notebook v9, sin cambios.
"""
import re
import unicodedata


# Catálogo peruano: entidades por sector, escenarios de fraude y escenarios legítimos.
# Es la "fuente única de verdad" del contexto peruano: la usan la adaptación del
# corpus internacional (2.4), la generación ampliada (2.5), el extractor de
# características (4) y el diagnóstico explicativo (6).


def normalizar(texto):
    """Minúsculas y sin tildes, para comparar palabras clave sin depender de acentos."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# --- Entidades peruanas o que operan en el Perú, agrupadas por sector ---
# Clave = nombre como aparece en un SMS; valor = alias "seguros" para detectarla en
# texto. Se evitan alias ambiguos (p. ej. "Claro" también es "claro que sí",
# "Prima" también es "mi prima", "Rímac" también es un distrito).
CATALOGO_ENTIDADES = {
    "banca": {
        "BCP": ['bcp', 'banco de credito', 'viabcp'],
        "Interbank": ['interbank'],
        "BBVA": ['bbva'],
        "Scotiabank": ['scotiabank'],
        "BanBif": ['banbif'],
        "Banco Pichincha": ['banco pichincha'],
        "Banco de la Nación": ['banco de la nacion'],
        "Mibanco": ['mibanco'],
        "Banco Falabella": ['banco falabella'],
        "Banco Ripley": ['banco ripley'],
        "Banco GNB": ['banco gnb'],
        "Banco de Comercio": ['banco de comercio'],
        "Citibank Perú": ['citibank peru'],
        "Alfin Banco": ['alfin banco'],
        "Banco Santander Perú": ['banco santander peru'],
        "Bank of China Perú": ['bank of china peru'],
        "ICBC Perú": ['icbc peru'],
        "Banco BCI Perú": ['banco bci peru'],
    },
    "cajas_financieras": {
        "Caja Arequipa": ['caja arequipa'],
        "Caja Huancayo": ['caja huancayo'],
        "Caja Piura": ['caja piura'],
        "Caja Cusco": ['caja cusco'],
        "Caja Trujillo": ['caja trujillo'],
        "Caja Sullana": ['caja sullana'],
        "Caja Ica": ['caja ica'],
        "Caja Tacna": ['caja tacna'],
        "Caja Maynas": ['caja maynas'],
        "Caja Metropolitana": ['caja metropolitana'],
        "Caja Los Andes": ['caja los andes'],
        "Compartamos Financiera": ['compartamos financiera'],
        "Financiera Confianza": ['financiera confianza'],
        "Financiera Oh!": ['financiera oh!'],
        "Financiera Efectiva": ['financiera efectiva'],
        "Financiera Qapaq": ['financiera qapaq'],
        "Crediscotia": ['crediscotia'],
        "Cooperativa Abaco": ['cooperativa abaco'],
        "Cooperativa Pacífico": ['cooperativa pacifico'],
        "Diners Club Perú": ['diners club peru'],
    },
    "billeteras_fintech": {
        "Yape": ['yape', 'yapeo', 'yapear', 'yapea', 'yapeame'],
        "Plin": ['plin'],
        "Tunki": ['tunki'],
        "BIM": ['bim'],
        "Ligo": ['ligo'],
        "Mercado Pago": ['mercado pago'],
        "Agora": ['agora'],
        "Kasnet": ['kasnet'],
        "Western Union": ['western union'],
        "Tpaga": ['tpaga'],
        "Culqi": ['culqi'],
        "Izipay": ['izipay'],
        "Niubiz": ['niubiz'],
        "Tarjeta Oh!": ['tarjeta oh!'],
        "Cuy Móvil": ['cuy movil'],
        "Rextie": ['rextie'],
        "Kambista": ['kambista'],
        "Binance": ['binance'],
        "Dólar House": ['dolar house'],
    },
    "estado": {
        "SUNAT": ['sunat'],
        "RENIEC": ['reniec'],
        "SAT Lima": ['sat lima'],
        "SUTRAN": ['sutran'],
        "MTC": ['mtc'],
        "Migraciones": ['migraciones'],
        "ONP": ['onp'],
        "EsSalud": ['essalud'],
        "SIS": ['sis'],
        "MINSA": ['minsa'],
        "Poder Judicial": ['poder judicial'],
        "Indecopi": ['indecopi'],
        "PNP": ['pnp'],
        "Ministerio de Trabajo": ['ministerio de trabajo'],
        "Municipalidad de Lima": ['municipalidad de lima'],
        "SBS": ['sbs'],
        "OSIPTEL": ['osiptel'],
        "OSINERGMIN": ['osinergmin'],
        "SUNARP": ['sunarp'],
        "SUNAFIL": ['sunafil'],
        "Ministerio Público": ['ministerio publico'],
        "Contraloría": ['contraloria'],
        "ONPE": ['onpe'],
        "JNE": ['jne'],
        "MIDIS": ['midis'],
        "Pensión 65": ['pension 65'],
        "Juntos": ['juntos'],
        "Qali Warma": ['qali warma'],
        "Fondo MiVivienda": ['fondo mivivienda'],
        "Techo Propio": ['techo propio'],
        "Ministerio de Educación": ['ministerio de educacion'],
        "SERVIR": ['servir'],
        "Defensoría del Pueblo": ['defensoria del pueblo'],
        "ATU": ['atu'],
        "SAT Callao": ['sat callao'],
        "Municipalidad de Surco": ['municipalidad de surco'],
        "Municipalidad de Miraflores": ['municipalidad de miraflores'],
        "Gobierno Regional de Arequipa": ['gobierno regional de arequipa'],
        "Senasa": ['senasa'],
        "Digesa": ['digesa'],
        "Sucamec": ['sucamec'],
        "Produce": ['produce'],
        "Bono Familiar Habitacional": ['bono familiar habitacional'],
        "Beca 18": ['beca 18'],
    },
    "telecom": {
        "Claro": ['claro peru', 'claro:', 'mi claro'],
        "Movistar": ['movistar'],
        "Entel": ['entel'],
        "Bitel": ['bitel'],
        "WOW Perú": ['wow peru'],
        "Win Internet": ['win internet'],
        "Mi Fibra": ['mi fibra'],
        "DirecTV": ['directv'],
    },
    "servicios": {
        "Luz del Sur": ['luz del sur'],
        "Enel": ['enel'],
        "Sedapal": ['sedapal'],
        "Cálidda": ['calidda'],
        "Hidrandina": ['hidrandina'],
        "Electrocentro": ['electrocentro'],
        "Electronoroeste": ['electronoroeste'],
        "Electro Sur Este": ['electro sur este'],
        "Sedalib": ['sedalib'],
        "Sedapar": ['sedapar'],
        "Epsel": ['epsel'],
        "Seal Arequipa": ['seal arequipa'],
        "Solgas": ['solgas'],
        "Lima Gas": ['lima gas'],
        "Zeta Gas": ['zeta gas'],
        "Petroperú": ['petroperu'],
        "Primax": ['primax'],
        "Repsol Perú": ['repsol peru'],
    },
    "courier_ecommerce": {
        "Serpost": ['serpost'],
        "Olva Courier": ['olva courier'],
        "Shalom": ['shalom'],
        "Urbano Express": ['urbano express'],
        "DHL": ['dhl'],
        "FedEx": ['fedex'],
        "Marvisur": ['marvisur'],
        "Cruz del Sur Cargo": ['cruz del sur cargo'],
        "Scharff": ['scharff'],
        "Chazki": ['chazki'],
        "Savar": ['savar'],
        "Mercado Libre": ['mercado libre'],
        "Falabella.com": ['falabella.com'],
        "Ripley": ['ripley'],
        "Oechsle": ['oechsle'],
        "Plaza Vea": ['plaza vea'],
        "Juntoz": ['juntoz'],
        "Linio": ['linio'],
        "AliExpress": ['aliexpress'],
        "Temu": ['temu'],
        "Shein": ['shein'],
        "Amazon": ['amazon'],
        "Rappi": ['rappi'],
        "PedidosYa": ['pedidosya'],
        "Didi Food": ['didi food'],
        "Aeropost": ['aeropost'],
        "Estafeta": ['estafeta'],
        "InkaFarma Delivery": ['inkafarma delivery'],
    },
    "afp_seguros": {
        "AFP Integra": ['afp integra'],
        "Prima AFP": ['prima afp'],
        "AFP Habitat": ['afp habitat'],
        "Profuturo AFP": ['profuturo afp'],
        "Rímac Seguros": ['rimac seguros'],
        "Pacífico Seguros": ['pacifico seguros'],
        "La Positiva": ['la positiva'],
        "Mapfre": ['mapfre'],
        "Interseguro": ['interseguro'],
        "Protecta": ['protecta'],
        "Chubb Perú": ['chubb peru'],
        "Qualitas": ['qualitas'],
        "SOAT": ['soat'],
        "Crecer Seguros": ['crecer seguros'],
        "Sanitas Perú": ['sanitas peru'],
        "Oncosalud": ['oncosalud'],
        "Auna": ['auna'],
    },
    "apps_suscripciones": {
        "Netflix": ['netflix'],
        "Spotify": ['spotify'],
        "Movistar TV": ['movistar tv'],
        "Disney+": ['disney+'],
        "Max": ['max'],
        "Prime Video": ['prime video'],
        "YouTube Premium": ['youtube premium'],
        "InDrive": ['indrive'],
        "Cabify": ['cabify'],
        "Uber": ['uber'],
        "Didi": ['didi'],
        "Beat": ['beat'],
        "Facebook": ['facebook'],
        "Instagram": ['instagram'],
        "TikTok": ['tiktok'],
        "Google": ['google'],
        "Apple ID": ['apple id'],
        "Microsoft": ['microsoft'],
        "Zoom": ['zoom'],
        "Steam": ['steam'],
        "Free Fire": ['free fire'],
        "Roblox": ['roblox'],
        "Tinder": ['tinder'],
    },
    "empleo_educacion": {
        "Computrabajo": ['computrabajo'],
        "Bumeran": ['bumeran'],
        "Laborum": ['laborum'],
        "LinkedIn": ['linkedin'],
        "Indeed": ['indeed'],
        "Pronabec": ['pronabec'],
        "SUNEDU": ['sunedu'],
        "PUCP": ['pucp'],
        "Universidad de Lima": ['universidad de lima'],
        "UPC": ['upc'],
        "USIL": ['usil'],
        "UTP": ['utp'],
        "UNI": ['uni'],
        "San Marcos": ['san marcos'],
        "Universidad Continental": ['universidad continental'],
        "Senati": ['senati'],
        "Cibertec": ['cibertec'],
        "Certus": ['certus'],
        "Toulouse Lautrec": ['toulouse lautrec'],
        "ICPNA": ['icpna'],
        "Británico": ['britanico'],
        "Intercorp Retail Empleos": ['intercorp retail empleos'],
    },
    "retail": {
        "Tottus": ['tottus'],
        "Wong": ['wong'],
        "Metro": ['metro'],
        "Inkafarma": ['inkafarma'],
        "Mifarma": ['mifarma'],
        "Boticas Perú": ['boticas peru'],
        "Promart": ['promart'],
        "Sodimac": ['sodimac'],
        "Maestro": ['maestro'],
        "Saga Falabella": ['saga falabella'],
        "Paris": ['paris'],
        "Real Plaza": ['real plaza'],
        "Jockey Plaza": ['jockey plaza'],
        "Mall Aventura": ['mall aventura'],
        "Estilos": ['estilos'],
        "Topitop": ['topitop'],
        "Hiraoka": ['hiraoka'],
        "Coolbox": ['coolbox'],
        "Efe": ['efe'],
        "La Curacao": ['la curacao'],
        "Curacao": ['curacao'],
        "Oeschle Tarjeta": ['oeschle tarjeta'],
        "Mass": ['mass'],
        "Tambo": ['tambo'],
        "Oxxo": ['oxxo'],
        "Makro": ['makro'],
        "Vivanda": ['vivanda'],
        "Plaza Norte": ['plaza norte'],
        "Cineplanet": ['cineplanet'],
        "Cinemark": ['cinemark'],
        "Bembos": ['bembos'],
        "KFC Perú": ['kfc peru'],
        "Pizza Hut Perú": ['pizza hut peru'],
        "Papa John's": ["papa john's"],
        "Starbucks Perú": ['starbucks peru'],
        "Norky's": ["norky's"],
        "Pardos Chicken": ['pardos chicken'],
        "Rokys": ['rokys'],
    },
    "transporte_viajes": {
        "LATAM": ['latam'],
        "Sky Airline": ['sky airline'],
        "JetSMART": ['jetsmart'],
        "Star Perú": ['star peru'],
        "Cruz del Sur": ['cruz del sur'],
        "Oltursa": ['oltursa'],
        "Civa": ['civa'],
        "Movil Bus": ['movil bus'],
        "Línea": ['linea'],
        "Excluciva": ['excluciva'],
        "PeruRail": ['perurail'],
        "Inca Rail": ['inca rail'],
        "Metropolitano": ['metropolitano'],
        "Línea 1": ['linea 1'],
        "Tarjeta Lima Pass": ['tarjeta lima pass'],
        "Aeropuerto Jorge Chávez": ['aeropuerto jorge chavez'],
        "Despegar": ['despegar'],
        "Booking": ['booking'],
        "Airbnb": ['airbnb'],
        "Peaje Rutas de Lima": ['peaje rutas de lima'],
        "Lima Expresa": ['lima expresa'],
        "Pase Vial": ['pase vial'],
    },
    "apuestas_entretenimiento": {
        "Te Apuesto": ['te apuesto'],
        "Inkabet": ['inkabet'],
        "Betsson": ['betsson'],
        "Doradobet": ['doradobet'],
        "Apuesta Total": ['apuesta total'],
        "Atlantic City": ['atlantic city'],
        "Teleticket": ['teleticket'],
        "Joinnus": ['joinnus'],
        "Ticketmaster Perú": ['ticketmaster peru'],
        "La Tinka": ['la tinka'],
        "Kábala": ['kabala'],
        "Gana Diario": ['gana diario'],
        "Intralot": ['intralot'],
    },
    "inmobiliario_autos": {
        "Urbania": ['urbania'],
        "Adondevivir": ['adondevivir'],
        "Nexo Inmobiliario": ['nexo inmobiliario'],
        "Los Portales": ['los portales'],
        "Menorca": ['menorca'],
        "Neoauto": ['neoauto'],
        "Autoland": ['autoland'],
        "Derco": ['derco'],
        "Toyota del Perú": ['toyota del peru'],
        "Maquisistema": ['maquisistema'],
        "Pandero": ['pandero'],
        "Autoplan": ['autoplan'],
    },
}

# Alias más específicos para nombres que también son palabras comunes del español
# ("línea", "juntos", "metro", "max"...): evita detectar entidades donde no las hay.
_ALIAS_ESPECIFICOS = {
    "Metro": ["supermercados metro", "metro:"], "Paris": ["tiendas paris"], "Línea": ["empresa linea", "transportes linea"],
    "Maestro": ["maestro home", "tiendas maestro"], "Mass": ["tiendas mass", "mass:"], "Max": ["hbo max"],
    "Juntos": ["programa juntos"], "Produce": ["ministerio de la produccion"], "Beat": ["beat taxi"],
    "Estilos": ["tiendas estilos"], "Efe": ["tiendas efe"], "Línea 1": ["linea 1 del metro"], "Uber": ["uber"],
    "Google": ["google", "gmail"], "Agora": ["agora pay", "agora:"], "Curacao": ["la curacao"],
    "Menorca": ["inmobiliaria menorca"], "Apple ID": ["apple id", "icloud"], "Didi": ["didi"],
    "Mi Fibra": ["mifibra"], "Win Internet": ["win internet", "win:"], "Mercado Pago": ["mercado pago"],
    "Crecer Seguros": ["crecer seguros"], "Qualitas": ["qualitas"], "Auna": ["auna"], "Juntoz": ["juntoz"],
    "SOAT": ["soat"], "Interseguro": ["interseguro"], "Protecta": ["protecta"], "La Tinka": ["la tinka", "tinka"],
    "Kábala": ["kabala"], "Gana Diario": ["gana diario"], "Oeschle Tarjeta": ["tarjeta oechsle"],
    "Cooperativa Pacífico": ["cooperativa pacifico"], "Pacífico Seguros": ["pacifico seguros"],
}

for _sector in CATALOGO_ENTIDADES.values():
    for _ent in _sector:
        if _ent in _ALIAS_ESPECIFICOS:
            _sector[_ent] = _ALIAS_ESPECIFICOS[_ent]

# Mapa plano entidad -> sector, útil para reportes.
SECTOR_DE_ENTIDAD = {ent: sector for sector, ents in CATALOGO_ENTIDADES.items() for ent in ents}

# Patrones de detección (con límites de palabra) de cada entidad.
_PATRONES_ENTIDAD = {
    ent: re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in alias) + r")(?![a-z0-9])")
    for sector in CATALOGO_ENTIDADES.values() for ent, alias in sector.items()
}


def detectar_entidades(texto):
    """Devuelve la lista de entidades del catálogo mencionadas en un SMS."""
    t = normalizar(texto)
    return [ent for ent, patron in _PATRONES_ENTIDAD.items() if patron.search(t)]


# --- Detección de URLs (una sola definición para todo el notebook) ---
# Dominios con TLD conocido (incluye acortadores como bit.ly, cutt.ly, wa.me, t.co)
# y enlaces con esquema o "www.". Se exige \b tras el TLD para no confundir texto
# normal ("pago.todo") con un enlace.
_TLDS = ("com|net|org|info|biz|xyz|top|online|click|site|link|app|live|shop|store|club|vip|website|"
         "space|lat|pe|ly|co|me|io|to|gl|cc|us|es|tk|ml|ga|cf|gq|cn|ru|icu|buzz|win|bid|loan|work|page")

REGEX_URL = re.compile(
    rf"(https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:{_TLDS})\b(?:/\S*)?)",
    re.IGNORECASE)

# v7: para el preprocesamiento y el diagnóstico, además de REGEX_URL se reconocen enlaces
# ofuscados como los de los SMISHING reales (//unusa.id/9rua, sqanzq.-com/cb81, http:/95.141.32.7,
# hxxp://...), que no terminan en una extensión conocida. REGEX_URL se sigue usando en el
# control de calidad de la sección 3, para que el corpus construido no cambie.
REGEX_ENLACE_AMPLIO = re.compile(
    REGEX_URL.pattern + r"|(?:hxxps?|https?):/*\S+|//\S+|\b[\w-]+(?:\.[\w-]+)*\.-?[a-z][\w-]*/\S*", re.IGNORECASE)

# --- Escenarios de SMISHING por gatillo psicológico (Cialdini, vía Shim et al., 2024) ---
# Cada escenario: (descripción, sectores de donde tomar la entidad, entidades concretas).
# Si se dan entidades concretas, se usan esas (p. ej. una papeleta solo la cobra SAT
# Lima, SUTRAN, el MTC o una municipalidad, nunca EsSalud).
ESCENARIOS_SMISHING = {
    "URGENCIA_MIEDO": [
        ("cuenta o tarjeta bloqueada por 'seguridad'", ["banca"], None),
        ("consumo o transferencia no reconocida que hay que 'anular' ya", ["banca", "billeteras_fintech"], None),
        ("token o clave digital desactivada, debe reactivarse hoy", ["banca", "billeteras_fintech"], None),
        ("actualización obligatoria de datos antes de que venza un plazo", ["banca", "billeteras_fintech", "afp_seguros"], None),
        ("intento de acceso desde otro dispositivo que se debe confirmar", ["banca", "billeteras_fintech", "apps_suscripciones"], None),
        ("línea móvil será suspendida por falta de validación de identidad", ["telecom"], None),
        ("corte de servicio por recibo impago", ["servicios", "telecom"], None),
        ("suscripción suspendida por problema con el pago", ["apps_suscripciones"], None),
        ("cuenta de red social o correo hackeada que hay que 'recuperar' ya", [], ["Facebook", "Instagram", "TikTok", "Google", "Apple ID", "Microsoft"]),
        ("SOAT o seguro vehicular vencido con riesgo de multa", [], ["SOAT", "Rímac Seguros", "Pacífico Seguros", "La Positiva", "Mapfre", "Interseguro"]),
        ("tarjeta de peaje o pase vial sin saldo, el vehículo será sancionado", [], ["Peaje Rutas de Lima", "Lima Expresa", "Pase Vial", "ATU"]),
        ("clave de cajero o de cajas municipales bloqueada", ["cajas_financieras"], None),
    ],
    "AUTORIDAD": [
        ("deuda tributaria o RUC observado con amenaza de cobranza coactiva", [], ["SUNAT"]),
        ("papeleta o multa de tránsito con descuento solo por hoy", [], ["SAT Lima", "SUTRAN", "MTC", "Municipalidad de Lima"]),
        ("DNI por caducar o bloqueo de identidad", [], ["RENIEC"]),
        ("citación judicial, denuncia o embargo de cuentas", [], ["Poder Judicial", "PNP"]),
        ("impedimento de salida del país o trámite migratorio observado", [], ["Migraciones"]),
        ("aportes previsionales observados o reclamo pendiente", [], ["ONP", "AFP Integra", "Prima AFP", "AFP Habitat", "Profuturo AFP"]),
        ("reclamo de consumidor o proceso sancionador pendiente", [], ["Indecopi"]),
        ("verificación obligatoria exigida por el regulador bancario", [], ["SBS", "BCP", "Interbank", "BBVA", "Scotiabank"]),
        ("partida registral o propiedad observada que debe regularizarse", [], ["SUNARP"]),
        ("multa electoral pendiente o actualización de datos electorales", [], ["ONPE", "JNE", "RENIEC"]),
        ("inspección laboral o sanción a la empresa del usuario", [], ["SUNAFIL", "Ministerio de Trabajo"]),
        ("línea o equipo será bloqueado por reporte de robo (IMEI)", [], ["OSIPTEL", "Claro", "Movistar", "Entel", "Bitel"]),
        ("cobranza extrajudicial de una deuda con central de riesgo", ["cajas_financieras", "banca"], None),
    ],
    "RECOMPENSA": [
        ("bono o subsidio del Estado pendiente de cobro", [], ["Banco de la Nación", "Ministerio de Trabajo", "Municipalidad de Lima"]),
        ("sorteo o premio ganado por usar un medio de pago", ["billeteras_fintech", "banca", "retail"], None),
        ("devolución de impuestos o reembolso a favor del usuario", [], ["SUNAT", "Mercado Libre", "Falabella.com", "Ripley", "Rappi"]),
        ("puntos, millas o saldo a favor por vencer", ["banca", "retail", "telecom"], None),
        ("retiro o liberación de fondos previsionales/CTS", [], ["AFP Integra", "Prima AFP", "AFP Habitat", "Profuturo AFP", "ONP"]),
        ("préstamo preaprobado sin requisitos", ["banca", "billeteras_fintech"], None),
        ("oferta de trabajo remoto con pago diario alto", [], ["Computrabajo", "Bumeran", "Mercado Libre", "Rappi"]),
        ("beca o beneficio educativo asignado", [], ["Pronabec", "Beca 18", "Ministerio de Educación"]),
        ("subsidio social o programa del Estado (vivienda, adulto mayor, alimentos)", [], ["Pensión 65", "Juntos", "Fondo MiVivienda", "Techo Propio", "MIDIS", "Bono Familiar Habitacional", "Qali Warma"]),
        ("premio de lotería, apuesta o casino online", ["apuestas_entretenimiento"], None),
        ("entradas gratis o sorteo de concierto/partido", [], ["Teleticket", "Joinnus", "Ticketmaster Perú", "Cineplanet", "Cinemark"]),
        ("pasajes o viaje gratis por aniversario de la empresa", ["transporte_viajes"], None),
        ("inversión en criptomonedas o cambio de dólares con ganancia garantizada", [], ["Binance", "Rextie", "Kambista", "Dólar House"]),
        ("vale de consumo o cupón de supermercado/restaurante", ["retail"], None),
        ("auto o casa sorteada en un programa de fondo colectivo", ["inmobiliario_autos"], None),
    ],
    "CONFIANZA_ENCOMIENDA": [
        ("paquete retenido en aduana con pago de arancel pendiente", [], ["Serpost", "DHL", "Olva Courier", "Mercado Libre"]),
        ("dirección incompleta, se debe reprogramar la entrega", ["courier_ecommerce"], None),
        ("pedido online con problema de pago o verificación", ["courier_ecommerce", "retail"], None),
        ("familiar o conocido que 'cambió de número' y pide un yapeo urgente", [], ["Yape", "Plin"]),
        ("comprador o vendedor de marketplace que pide confirmar con un código", [], ["Mercado Libre", "Yape", "Plin", "Mercado Pago"]),
        ("soporte técnico que pide el código de verificación recibido", ["apps_suscripciones", "billeteras_fintech", "telecom"], None),
        ("alquiler de departamento o cuarto que pide adelanto por Yape", [], ["Urbania", "Adondevivir", "Airbnb", "Yape"]),
        ("reserva de hotel, pasaje o bus que debe 'confirmarse' pagando", ["transporte_viajes"], None),
        ("universidad o instituto con matrícula/pensión pendiente", ["empleo_educacion"], None),
        ("delivery de comida o farmacia con cobro pendiente", [], ["Rappi", "PedidosYa", "Didi Food", "InkaFarma Delivery", "Bembos", "KFC Perú"]),
        ("taxi por aplicativo: 'olvidaste un objeto' o cobro extra", [], ["InDrive", "Uber", "Didi", "Cabify"]),
        ("gas a domicilio o balón de gas con promoción que exige pago adelantado", [], ["Solgas", "Lima Gas", "Zeta Gas"]),
    ],
}

# Cómo "llama a la acción" el estafador. Varía para que el modelo no aprenda que
# smishing == tiene URL (hay estafas sin enlace: llamar, escribir a WhatsApp, etc.).
CANALES_SMISHING = {
    "url": "incluye un enlace corto ficticio y verosímil (dominio inventado, no uno real)",
    "telefono": "en vez de enlace, pide llamar a un número de teléfono peruano ficticio",
    "whatsapp": "en vez de enlace, pide escribir a un WhatsApp ficticio (+51 9XX XXX XXX)",
    "responder": "pide responder el SMS con un código, DNI o dato personal, sin enlaces",
}

# Estilos de redacción del SMISHING: el fraude real no siempre es formal ni tiene
# faltas; también se escribe informal y con jerga (p. ej. el "familiar" o el vendedor).
ESTILOS = {
    "corporativo": "tono formal que imita el lenguaje corporativo de la entidad",
    "alarmista": "tono alarmista, con alguna palabra en MAYÚSCULAS",
    "descuidado": "redacción informal con alguna falta de ortografía o sin tildes",
    "informal_peruano": "tono cercano e informal con jerga peruana (como si escribiera un conocido o un vendedor)",
    "breve": "muy corto, menos de 90 caracteres",
}

# Estilos del HAM: también hay HAM muy corto y HAM largo, para que la longitud no
# sea un atajo entre clases.
ESTILOS_HAM = {
    "natural": "longitud natural para ese tipo de mensaje",
    "breve": "muy corto, menos de 60 caracteres",
    "detallado": "más largo, cerca de 160 caracteres",
}

# --- Escenarios de HAM (mensajes legítimos) ---
# Incluyen notificaciones REALES de las mismas entidades (hard negatives): así el
# modelo aprende que mencionar al BCP o a SUNAT no es, por sí solo, fraude.
ESCENARIOS_HAM = {
    "notificacion_legitima": [
        ("confirmación de un consumo con tarjeta, sin pedir datos ni enlaces", ["banca"], None),
        ("constancia de yapeo o plineo recibido o enviado", [], ["Yape", "Plin", "Tunki", "Mercado Pago"]),
        ("código de verificación legítimo que advierte 'no lo compartas con nadie'", ["banca", "billeteras_fintech", "apps_suscripciones"], None),
        ("recordatorio de pago de cuota o recibo con fecha, sin amenazas", ["banca", "servicios", "telecom"], None),
        ("estado de un pedido o entrega exitosa", ["courier_ecommerce", "retail"], None),
        ("recordatorio de cita médica", [], ["EsSalud", "SIS", "MINSA"]),
        ("confirmación de recarga o cambio de plan", ["telecom"], None),
        ("confirmación de un trámite presentado, sin pedir datos", [], ["SUNAT", "RENIEC", "ONP", "Migraciones", "AFP Integra", "Prima AFP", "SUNARP"]),
        ("confirmación de compra de pasaje o boleto de bus", ["transporte_viajes"], None),
        ("aviso de corte programado de luz o agua por mantenimiento", ["servicios"], None),
        ("constancia de pago de pensión universitaria o matrícula", ["empleo_educacion"], None),
        ("recordatorio de vencimiento de SOAT o revisión técnica, sin enlaces sospechosos", [], ["SOAT", "Rímac Seguros", "Pacífico Seguros", "La Positiva"]),
        ("confirmación de desembolso de préstamo solicitado en agencia", ["cajas_financieras", "banca"], None),
        ("código de viaje o llegada del conductor de taxi por aplicativo", [], ["InDrive", "Uber", "Didi", "Cabify"]),
        ("estado de pedido de delivery de comida", [], ["Rappi", "PedidosYa", "Didi Food", "Bembos", "Pardos Chicken", "Norky's"]),
        ("confirmación de entradas compradas", ["apuestas_entretenimiento"], None),
    ],
    "promocional_legitimo": [
        ("promoción real de tienda o supermercado, sin pedir datos personales", ["retail", "courier_ecommerce"], None),
        ("campaña de descuento de la app o del plan móvil", ["telecom", "apps_suscripciones"], None),
        ("promoción de cine, restaurante o delivery", [], ["Cineplanet", "Cinemark", "Bembos", "KFC Perú", "Pizza Hut Perú", "Rokys", "Rappi"]),
        ("campaña de matrícula o curso de universidad/instituto", ["empleo_educacion"], None),
        ("oferta de pasajes o paquetes turísticos", ["transporte_viajes"], None),
    ],
    "conversacion": [
        ("coordinación familiar (almuerzo del domingo, recoger a alguien)", [], None),
        ("conversación entre amigos usando jerga peruana (pe, causa, oe, al toque, chamba, jato)", [], None),
        ("coordinación de trabajo u oficina", [], None),
        ("conversación universitaria (trabajo grupal, clases, exámenes)", [], None),
        ("coordinación de una pichanga, pollada o colecta, incluyendo yapeos entre personas", [], ["Yape", "Plin"]),
        ("compra o venta entre conocidos, sin presión ni enlaces", [], None),
        ("mensajes del grupo de vecinos o del condominio", [], None),
        ("coordinación de viaje a provincia (Cusco, Arequipa, Piura) o de visita a la familia", [], None),
        ("conversación sobre el partido de la selección, la U, Alianza o Cristal", [], None),
        ("pedido o venta de un emprendimiento pequeño (tortas, ropa, comida)", [], ["Yape", "Plin"]),
        ("aviso del colegio o de la profesora a los padres de familia", [], None),
        ("coordinación de una junta, cumpleaños o parrillada", [], None),
        ("mensajes de pareja o amigos cercanos, cariñosos y cotidianos", [], None),
    ],
}

# Detalles "hiperlocales" para que los mensajes suenen peruanos de verdad.
LUGARES_PERU = ["Miraflores", "San Juan de Lurigancho", "Surco", "Los Olivos", "Comas", "Callao",
                "Villa El Salvador", "San Miguel", "Arequipa", "Trujillo", "Chiclayo", "Piura",
                "Cusco", "Huancayo", "Iquitos", "Ate", "Chorrillos", "La Molina", "Breña", "Independencia",
                "San Juan de Miraflores", "Villa María del Triunfo", "Puente Piedra", "Carabayllo", "Lince",
                "Jesús María", "Pueblo Libre", "San Borja", "Magdalena", "Barranco", "Rímac", "La Victoria",
                "Gamarra", "Mesa Redonda", "Ica", "Tacna", "Puno", "Juliaca", "Cajamarca", "Ayacucho",
                "Huánuco", "Pucallpa", "Tarapoto", "Tumbes", "Chimbote", "Huaraz", "Moquegua", "Abancay",
                "Puerto Maldonado", "Chachapoyas", "Sullana", "Huacho", "Cañete", "Chincha", "Ilo"]

GATILLOS = list(ESCENARIOS_SMISHING)

_todas = set(SECTOR_DE_ENTIDAD)

for _esc in list(ESCENARIOS_SMISHING.values()) + list(ESCENARIOS_HAM.values()):
    for _desc, _sect, _ents in _esc:
        assert all(s in CATALOGO_ENTIDADES for s in _sect), _desc
        assert _ents is None or all(e in _todas for e in _ents), (_desc, _ents)
