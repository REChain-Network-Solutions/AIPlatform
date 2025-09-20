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

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель CurrencyCode.
 *
 * Трехбуквенный код валюты в формате [ISO-4217](https://www.iso.org/iso-4217-currency-codes.html).
 *
 * Пример: ~`RUB`. Должен соответствовать валюте субаккаунта (`recipient.gateway_id`),
 * если вы разделяете потоки платежей, и валюте аккаунта (shopId в [личном кабинете](https://yookassa.ru/my)), если не разделяете.
 *
 * @author  cms@yoomoney.ru
 */
class CurrencyCode extends AbstractEnum
{
    /** Российский рубль */
    public const RUB = 'RUB';

    /** Доллар США */
    public const USD = 'USD';

    /** Евро */
    public const EUR = 'EUR';

    /** Белорусский рубль */
    public const BYN = 'BYN';

    /** Китайская йена */
    public const CNY = 'CNY';

    /** Казахский тенге */
    public const KZT = 'KZT';

    /** Украинская гривна */
    public const UAH = 'UAH';

    /** Узбекский сум */
    public const UZS = 'UZS';

    /** Турецкая лира */
    public const _TRY = 'TRY';

    /** Индийская рупия */
    public const INR = 'INR';

    /** Молдавский лей */
    public const MDL = 'MDL';

    /** Азербайджанский манат */
    public const AZN = 'AZN';

    /** Армянский драм */
    public const AMD = 'AMD';

    protected static array $validValues = [
        self::RUB => true,
        self::USD => true,
        self::EUR => true,
        self::BYN => true,
        self::CNY => true,
        self::KZT => true,
        self::UAH => true,
        self::UZS => true,
        self::_TRY => true,
        self::INR => true,
        self::MDL => true,
        self::AZN => true,
        self::AMD => true,
    ];
}
