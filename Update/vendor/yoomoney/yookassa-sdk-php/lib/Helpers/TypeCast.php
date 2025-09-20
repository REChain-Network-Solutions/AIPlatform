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

namespace YooKassa\Helpers;

use DateTime;

/**
 * Класс, представляющий модель TypeCast.
 *
 * Класс для преобразования типов значений.
 *
 * @category Class
 * @package  YooKassa\Helpers
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class TypeCast
{
    /**
     * Проверяет, может ли переданное значение быть преобразовано в строку.
     *
     * @param mixed $value Проверяемое значение
     *
     * @return bool True если значение преобразовать в строку можно, false если нет
     */
    public static function canCastToString(mixed $value): bool
    {
        if (is_scalar($value)) {
            return !is_bool($value) && !is_resource($value);
        }
        if (is_object($value)) {
            return method_exists($value, '__toString');
        }

        return false;
    }

    /**
     * Проверяет можно ли преобразовать переданное значение в строку из перечисления.
     *
     * @param mixed $value Проверяемое значение
     *
     * @return bool True если значение преобразовать в строку можно, false если нет
     */
    public static function canCastToEnumString(mixed $value): bool
    {
        if (is_string($value) && '' !== $value) {
            return true;
        }
        if (is_object($value)) {
            return method_exists($value, '__toString');
        }

        return false;
    }

    /**
     * Проверяет, можно ли преобразовать переданное значение в объект даты-времени.
     *
     * @param mixed $value Провеяремое значение
     *
     * @return bool True если значение можно преобразовать в объект даты, false если нет
     */
    public static function canCastToDateTime(mixed $value): bool
    {
        if ($value instanceof DateTime) {
            return true;
        }
        if (is_numeric($value)) {
            $value = (float) $value;

            return $value >= 0;
        }
        if (is_string($value)) {
            return '' !== $value;
        }
        if (is_object($value)) {
            return method_exists($value, '__toString') && ((string) $value) !== '';
        }

        return false;
    }

    /**
     * Преобразует переданне значение в объект типа DateTime.
     *
     * @param DateTime|int|string $value Преобразуемое значение
     *
     * @return null|DateTime Объект типа DateTime или null, если при парсинг даты не удался
     */
    public static function castToDateTime(mixed $value): ?DateTime
    {
        if ($value instanceof DateTime) {
            return clone $value;
        }
        if (is_numeric($value)) {
            $date = new DateTime();
            $date->setTimestamp((int) $value);
        } elseif (is_string($value) || (is_object($value) && method_exists($value, '__toString'))) {
            $date = date_create((string) $value);
            if (false === $date) {
                $date = null;
            }
        } else {
            $date = null;
        }

        return $date;
    }

    /**
     * Проверяет можно ли преобразовать переданное значение в буллево значение.
     *
     * @param mixed $value Проверяемое значение
     *
     * @return bool True если значение преобразуется в bool, false если нет
     */
    public static function canCastToBoolean(mixed $value): bool
    {
        return is_numeric($value) || is_bool($value);
    }
}
