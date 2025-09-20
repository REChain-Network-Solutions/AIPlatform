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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentPeriod.
 *
 * Период оплаты, за который выставлены начисления и за который вносится оплата.
 * 
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property int $month Месяц периода. Например, ~`1` — январь.
 * @property int $year Год периода. Например, ~`2025`.
*/
class PaymentPeriod extends AbstractObject 
{
    /**
     * Месяц периода. Например, ~`1` — январь.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    #[Assert\LessThanOrEqual(12)]
    private ?int $_month = null;

    /**
     * Год периода. Например, ~`2025`.
     *
     * @var int|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    private ?int $_year = null;

    /**
     * Возвращает month.
     *
     * @return int|null
     */
    public function getMonth(): ?int
    {
        return $this->_month;
    }

    /**
     * Устанавливает month.
     *
     * @param int|null $month Месяц периода. Например, ~`1` — январь.
     *
     * @return self
     */
    public function setMonth(?int $month = null): self
    {
        $this->_month = $this->validatePropertyValue('_month', $month);
        return $this;
    }

    /**
     * Возвращает year.
     *
     * @return int|null
     */
    public function getYear(): ?int
    {
        return $this->_year;
    }

    /**
     * Устанавливает year.
     *
     * @param int|null $year Год периода. Например, ~`2025`.
     *
     * @return self
     */
    public function setYear(?int $year = null): self
    {
        $this->_year = $this->validatePropertyValue('_year', $year);
        return $this;
    }
}

