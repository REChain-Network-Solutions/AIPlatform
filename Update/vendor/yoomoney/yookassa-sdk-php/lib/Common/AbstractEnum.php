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

namespace YooKassa\Common;

/**
 * Класс, представляющий модель AbstractEnum.
 *
 * Базовый класс генерируемых enum'ов.
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
abstract class AbstractEnum
{
    /**
     * @var array Массив принимаемых enum'ом значений
     */
    protected static array $validValues = [];

    /**
     * Проверяет наличие значения в enum'e.
     *
     * @param mixed $value Проверяемое значение
     *
     * @return bool True если значение имеется, false если нет
     */
    public static function valueExists(mixed $value): bool
    {
        return array_key_exists($value, static::$validValues);
    }

    /**
     * Возвращает все значения в enum'e.
     *
     * @return array Массив значений в перечислении
     */
    public static function getValidValues(): array
    {
        return array_keys(static::$validValues);
    }

    /**
     * Возвращает значения в enum'е значения которых разрешены.
     *
     * @return string[] Массив разрешённых значений
     */
    public static function getEnabledValues(): array
    {
        return array_keys(array_filter(static::$validValues, static fn ($v) => $v, ARRAY_FILTER_USE_BOTH));
    }
}
