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

namespace YooKassa\Model\Payment\PaymentMethod;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель BankCardType.
 *
 * Тип банковской карты.
 *
 * Возможные значения:
 * - `MasterCard` (для карт Mastercard и Maestro),
 * - `Visa` (для карт Visa и Visa Electron),
 * - `Mir`,
 * - `UnionPay`,
 * - `JCB`,
 * - `AmericanExpress`,
 * - `DinersClub`,
 * - `DiscoverCard`,
 * - `InstaPayment`,
 * - `InstaPaymentTM`,
 * - `Laser`,
 * - `Dankort`,
 * - `Solo`,
 * - `Switch`,
 * - `Unknown`.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class BankCardType extends AbstractEnum
{
    public const MASTER_CARD = 'MasterCard';
    public const VISA = 'Visa';
    public const MIR = 'Mir';
    public const UNION_PAY = 'UnionPay';
    public const JCB = 'JCB';
    public const AMERICAN_EXPRESS = 'AmericanExpress';
    public const DINERS_CLUB = 'DinersClub';
    public const DISCOVER_CARD_CLUB = 'DiscoverCard';
    public const INSTA_PAYMENT_CLUB = 'InstaPayment';
    public const INSTA_PAYMENT_TM_CLUB = 'InstaPaymentTM';
    public const LASER_CLUB = 'Laser';
    public const DANKORT_CLUB = 'Dankort';
    public const SOLO_CLUB = 'Solo';
    public const SWITCH_CLUB = 'Switch';
    public const UNKNOWN = 'Unknown';

    protected static array $validValues = [
        self::MASTER_CARD => true,
        self::VISA => true,
        self::MIR => true,
        self::UNION_PAY => true,
        self::JCB => true,
        self::AMERICAN_EXPRESS => true,
        self::DINERS_CLUB => true,
        self::DISCOVER_CARD_CLUB => true,
        self::INSTA_PAYMENT_CLUB => true,
        self::INSTA_PAYMENT_TM_CLUB => true,
        self::LASER_CLUB => true,
        self::DANKORT_CLUB => true,
        self::SOLO_CLUB => true,
        self::SWITCH_CLUB => true,
        self::UNKNOWN => true,
    ];
}
