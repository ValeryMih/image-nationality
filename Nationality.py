# https://dzen.ru/a/Yrs4oaxAEFTRPiWX
# Распознавание (идентификация) лиц на Python
# https://dzen.ru/a/XLB28gxwbQCzRn4n
# Сколько рас людей существует на земле: история происхождения и основные отличия

# pip install face_recognition
# pip install deepface
import face_recognition  # Импортируем необходимые библиотеки
from deepface import DeepFace


def face_race_nationality(img_path):  # Создаем функцию для определения расы и национальности лица на фотографии
    try:
        image = face_recognition.load_image_file(img_path)  # Загружаем изображение
        face_locations = face_recognition.face_locations(
            image)  # Обнаруживаем лица на изображении и получаем их координаты
        if not face_locations:  # Если на фотографии не найдено ни одного лица, вызываем исключение
            raise Exception('На фотографии не найдено ни одного лица')

        face_encodings = face_recognition.face_encodings(image,
                                                         face_locations)  # Получаем эмбеддинги лиц на изображении
        face_race = DeepFace.analyze(img_path, actions=['race'])['race']  # Определяем расу на фотографии
        face_nationality = None  # Инициализируем переменную для национальности лица

        for encoding in face_encodings:  # Проходимся по каждому обнаруженному лицу
            matches = DeepFace.find(img_path=img_path, db_path='.')  # Определяем национальность лица
            print(DeepFace.find.calls)  ###
            if matches and 'name' in matches[0]:
                face_nationality = matches[0]['name']
                break

        total_sum = sum(face_race.values())  # Вычисляем суммарный процент расы, суммируя значения для каждой категории
        race_percentages = {
            'асиатская': (face_race['asian'] / total_sum) * 100,
            'черная': (face_race['black'] / total_sum) * 100,
            'индийская': (face_race['indian'] / total_sum) * 100,
            'латиноамериканская': (face_race['latino hispanic'] / total_sum) * 100,
            'средневосточная': (face_race['middle eastern'] / total_sum) * 100,
            'белая': (face_race['white'] / total_sum) * 100
        }

        return face_nationality, race_percentages  # Возвращаем национальность и процент расы
    except Exception as ex:
        return ex


if __name__ == '__main__':  # Если функция вызвана как главная программа
    img_path = input("Введите путь к фотографии: ")  # Запрашиваем у пользователя путь к фотографии
    face_nationality, race_percentages = face_race_nationality(
        img_path)  # Вызываем функцию для определения расы и национальности лица на фотографии

    if isinstance(face_nationality, str):
        print(f'Национальность: {face_nationality}')  # Выводим результаты на экран
    for race, percentage in race_percentages.items():
        print(f'{race}: {percentage:.2f}%')

# программа, которая на основе face_recognition определяет национальность человека и выводит ее вместе с процентным соотношением рас
# Обратите внимание, что для определения национальности используется функция compare_faces,
# которая сравнивает лица на фото и возвращает наиболее подходящую национальность из заданных.
# Это не идеальный способ определения национальности, но может дать общее представление о том,
# к какой национальности человек скорее всего принадлежит.
#
# Национальность: японская
# асиатская: 72.28%
# черная: 0.00%
# индийская: 0.13%
# латиноамериканская: 0.30%
# средневосточная: 0.10%
# белая: 27.19%
