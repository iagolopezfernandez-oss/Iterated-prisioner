from openai import OpenAI
import sys
from pathlib import Path

def cargar_api_key(nombre_archivo="api_key"):
    ruta = Path(nombre_archivo)

    if not ruta.exists():
        print(f"❌ No se encuentra el archivo '{nombre_archivo}'")
        sys.exit(1)

    api_key = ruta.read_text(encoding="utf-8").strip()

    if not api_key:
        print("❌ El archivo 'api_key' está vacío")
        sys.exit(1)

    return api_key


def main():
    api_key = cargar_api_key()

    client = OpenAI(api_key=api_key)

    try:
        modelos = client.models.list()
    except Exception as e:
        print("❌ Error al consultar los modelos:")
        print(e)
        sys.exit(1)

    print("\n✅ Modelos disponibles para esta API key:\n")

    for modelo in sorted(modelos.data, key=lambda m: m.id):
        print(modelo.id)


if __name__ == "__main__":
    main()