<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Receipt;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель ReceiptItemMeasure.
 *
 * Мера количества предмета расчета передается в массиве `items`, в параметре `measure`.
 *
 * Обязательный параметр, если используете Чеки от ЮKassa или онлайн-кассу, обновленную до ФФД 1.2.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class ReceiptItemMeasure extends AbstractEnum
{
    /** Штука, единица товара */
    public const PIECE = 'piece';

    /** Грамм */
    public const GRAM = 'gram';

    /** Килограмм */
    public const KILOGRAM = 'kilogram';

    /** Тонна */
    public const TON = 'ton';

    /** Сантиметр */
    public const CENTIMETER = 'centimeter';

    /** Дециметр */
    public const DECIMETER = 'decimeter';

    /** Метр */
    public const METER = 'meter';

    /** Квадратный сантиметр */
    public const SQUARE_CENTIMETER = 'square_centimeter';

    /** Квадратный дециметр */
    public const SQUARE_DECIMETER = 'square_decimeter';

    /** Квадратный метр */
    public const SQUARE_METER = 'square_meter';

    /** Миллилитр */
    public const MILLILITER = 'milliliter';

    /** Литр */
    public const LITER = 'liter';

    /** Кубический метр */
    public const CUBIC_METER = 'cubic_meter';

    /** Килловат-час */
    public const KILOWATT_HOUR = 'kilowatt_hour';

    /** Гигакалория */
    public const GIGACALORIE = 'gigacalorie';

    /** Сутки */
    public const DAY = 'day';

    /** Час */
    public const HOUR = 'hour';

    /** Минута */
    public const MINUTE = 'minute';

    /** Секунда */
    public const SECOND = 'second';

    /** Килобайт */
    public const KILOBYTE = 'kilobyte';

    /** Мегабайт */
    public const MEGABYTE = 'megabyte';

    /** Гигабайт */
    public const GIGABYTE = 'gigabyte';

    /** Терабайт */
    public const TERABYTE = 'terabyte';

    /** Другое */
    public const ANOTHER = 'another';

    protected static array $validValues = [
        self::PIECE => true,
        self::GRAM => true,
        self::KILOGRAM => true,
        self::TON => true,
        self::CENTIMETER => true,
        self::DECIMETER => true,
        self::METER => true,
        self::SQUARE_CENTIMETER => true,
        self::SQUARE_DECIMETER => true,
        self::SQUARE_METER => true,
        self::MILLILITER => true,
        self::LITER => true,
        self::CUBIC_METER => true,
        self::KILOWATT_HOUR => true,
        self::GIGACALORIE => true,
        self::DAY => true,
        self::HOUR => true,
        self::MINUTE => true,
        self::SECOND => true,
        self::KILOBYTE => true,
        self::MEGABYTE => true,
        self::GIGABYTE => true,
        self::TERABYTE => true,
        self::ANOTHER => true,
    ];
}
