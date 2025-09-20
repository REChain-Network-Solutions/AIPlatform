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

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Class MarkQuantity.
 *
 * Дробное количество маркированного товара (тег в 54 ФЗ — 1291).
 * Обязательный параметр, если одновременно выполняются эти условия:
 * * вы используете Чеки от ЮKassa или онлайн-кассу, обновленную до ФФД 1.2;
 * * товар нужно [маркировать](http://docs.cntd.ru/document/902192509);
 * * поле `measure` имеет значение ~`piece`.
 *
 * Пример: вы продаете поштучно карандаши.
 * Они поставляются пачками по 100 штук с одним кодом маркировки.
 * При продаже одного карандаша нужно в `numerator` передать ~`1`, а в `denominator` — ~`100`.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $numerator Числитель — количество продаваемых товаров из одной потребительской упаковки (тег в 54 ФЗ — 1293)
 * @property string $denominator Знаменатель — общее количество товаров в потребительской упаковке (тег в 54 ФЗ — 1294)
 */
class MarkQuantity extends AbstractObject
{
    /** @var int Минимальное значение */
    public const MIN_VALUE = 1;

    /**
     * @var int|null Числитель — количество продаваемых товаров из одной потребительской упаковки (тег в 54 ФЗ — 1293). Не может превышать denominator
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    private ?int $_numerator = null;

    /**
     * @var int|null Знаменатель — общее количество товаров в потребительской упаковке (тег в 54 ФЗ — 1294)
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    private ?int $_denominator = null;

    /**
     * Возвращает числитель.
     *
     * @return int|null Числитель
     */
    public function getNumerator(): ?int
    {
        return $this->_numerator;
    }

    /**
     * Устанавливает числитель.
     *
     * @param int|null $numerator Числитель
     *
     * @return self
     */
    public function setNumerator(?int $numerator = null): self
    {
        $this->_numerator = $this->validatePropertyValue('_numerator', $numerator);
        return $this;
    }

    /**
     * Возвращает знаменатель.
     *
     * @return int|null Знаменатель
     */
    public function getDenominator(): ?int
    {
        return $this->_denominator;
    }

    /**
     * Устанавливает знаменатель.
     *
     * @param int|null $denominator Знаменатель
     *
     * @return self
     */
    public function setDenominator(?int $denominator = null): self
    {
        $this->_denominator = $this->validatePropertyValue('_denominator', $denominator);
        return $this;
    }
}
