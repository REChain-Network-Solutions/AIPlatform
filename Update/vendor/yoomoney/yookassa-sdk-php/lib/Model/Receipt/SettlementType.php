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
 * SettlementType - Тип расчета.
 *
 * Тип расчета передается в запросе на создание чека в массиве `settlements`, в параметре `type`.
 *
 * Возможные значения:
 * - `cashless` - Безналичный расчет
 * - `prepayment` - Предоплата (аванс)
 * - `postpayment` - Постоплата (кредит)
 * - `consideration` - Встречное предоставление
 */
class SettlementType extends AbstractEnum
{
    /** Безналичный расчет */
    public const CASHLESS = 'cashless';

    /** Предоплата (аванс) */
    public const PREPAYMENT = 'prepayment';

    /** Постоплата (кредит) */
    public const POSTPAYMENT = 'postpayment';

    /** Встречное предоставление */
    public const CONSIDERATION = 'consideration';

    protected static array $validValues = [
        self::CASHLESS => true,
        self::PREPAYMENT => true,
        self::POSTPAYMENT => true,
        self::CONSIDERATION => true,
    ];
}
