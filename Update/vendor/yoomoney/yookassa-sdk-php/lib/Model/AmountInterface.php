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

namespace YooKassa\Model;

/**
 * Interface AmountInterface.
 *
 * @property string $value Сумма
 * @property string $currency Код валюты
 */
interface AmountInterface
{
    /**
     * Возвращает значение суммы.
     *
     * @return string Сумма
     */
    public function getValue(): string;

    /**
     * Устанавливает значение суммы.
     *
     * @param numeric|string $value Сумма
     */
    public function setValue(mixed $value);

    /**
     * Возвращает сумму в копейках в виде целого числа.
     *
     * @return int Сумма в копейках/центах
     */
    public function getIntegerValue(): int;

    /**
     * Возвращает валюту.
     *
     * @return string Код валюты
     */
    public function getCurrency(): string;

    /**
     * Устанавливает код валюты.
     *
     * @param string $currency Код валюты
     */
    public function setCurrency(string $currency);

    /**
     * Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.
     *
     * @return array Ассоциативный массив со свойствами текущего объекта
     */
    public function toArray(): array;
}
