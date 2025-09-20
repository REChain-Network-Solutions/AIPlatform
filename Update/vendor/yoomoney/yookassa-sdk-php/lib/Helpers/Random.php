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

use Exception;

/**
 * Класс, представляющий модель Random.
 *
 * Класс хэлпера для генерации случайных значений, используется в тестах.
 *
 * @category Class
 * @package  YooKassa\Helpers
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class Random
{
    /**
     * Возвращает рандомное целое число. По умолчанию возвращает число от нуля до PHP_INT_MAX.
     *
     * @param null|int $min Минимально возможное значение
     * @param null|int $max Максимально возможное значение
     * @return int Рандомное целое число
     */
    public static function int(?int $min = null, ?int $max = null): int
    {
        if (null === $min) {
            $min = 0;
        }
        if (null === $max) {
            $max = PHP_INT_MAX;
        }
        try {
            return random_int($min, $max);
        } catch (Exception) {
            return $min;
        }
    }

    /**
     * Возвращает рандомное число с плавающей точкой. По умолчанию возвращает значение в промежутке от нуля до едениы.
     *
     * @param null|float $min Минимально возможное значение
     * @param null|float $max Максимально возможное значение
     *
     * @return float Рандомное число с плавающей точкой
     */
    public static function float(?float $min = null, ?float $max = null): float
    {
        $random = self::int() / PHP_INT_MAX;
        if (null === $min) {
            $min = 0.0;
        }
        if (null === $max) {
            return $random + $min;
        }

        return ($random * ($max - $min)) + $min;
    }

    /**
     * Возвращает строку из рандомных символов.
     *
     * @param int $length Длина возвращаемой строки, или минимальная длина, если передан парамтр $maxLength
     * @param null|int|string $maxLength Если параметр не равен null, возвращает сроку длиной от $length до $maxLength
     * @param null|array|string $characters Строка или массив используемых в строке символов
     *
     * @return string Строка, состоящая из рандомных символов
     */
    public static function str(int $length, mixed $maxLength = null, array|string|null $characters = null): string
    {
        $result = '';
        if (null !== $maxLength) {
            if (is_string($maxLength)) {
                $characters = $maxLength;
            } else {
                $length = self::int($length, $maxLength);
            }
        }
        if ($characters === null) {
            for ($i = 0; $i < $length; $i++) {
                $chr = chr(self::int(32, 125));
                $result .= $chr;
            }
        } else {
            for ($i = 0; $i < $length; $i++) {
                $chr = mb_substr($characters, self::int(0, mb_strlen($characters) - 1), 1);
                $result .= $chr;
            }
        }

        return $result;
    }

    /**
     * Возвращает строку, состоящую из символов '0123456789abcdef'.
     *
     * @param int $length Длина возвращаемой строки
     * @param bool $useBest Использовать ли функцию random_int если она доступна
     *
     * @return string Строка, состоящая из рандомных символов
     */
    public static function hex(int $length, bool $useBest = true): string
    {
        return self::str($length, '0123456789abcdef', $useBest);
    }

    /**
     * Возвращает рандомную последовательность байт
     *
     * @param int $length Длина возвращаемой строки
     *
     * @return string Строка, состоящая из рандомных символов
     */
    public static function bytes(int $length): string
    {
        $result = '';
        try {
            $result = random_bytes($length);
        } catch (Exception) {
            for ($i = 0; $i < $length; $i++) {
                $chr = chr(self::int(0, 255));
                $result .= $chr;
            }
        }

        return $result;
    }

    /**
     * Возвращает рандомное значение из переданного массива.
     *
     * @param array $values Массив источник данных
     *
     * @return mixed Случайное значение из переданного массива
     */
    public static function value(array $values): mixed
    {
        return $values[self::int(0, count($values) - 1)];
    }

    /**
     * Возвращает рандомное буллево значение.
     *
     * @return bool Либо true либо false, одно из двух
     */
    public static function bool(): bool
    {
        return 1 === self::int(0, 1);
    }
}
