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

namespace YooKassa\Model\Payment;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель PaymentMethodType.
 *
 * Код способа оплаты — тип платежного средства, которое используется для оплаты. [Подробнее о способах оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-methods)
 *
 * Возможные значения:
 * - `yoo_money` - Платеж из кошелька ЮMoney
 * - `bank_card` - Платеж с произвольной банковской карты
 * - `sberbank` - Платеж СбербанкОнлайн
 * - `cash` - Платеж наличными
 * - `mobile_balance` - Платеж с баланса мобильного телефона
 * - `apple_pay` - Платеж ApplePay
 * - `google_pay` - Платеж Google Pay
 * - `qiwi` - Платеж из кошелька Qiwi
 * - `webmoney` - Платеж из кошелька Webmoney
 * - `alfabank` - Платеж через Альфа-Клик
 * - `b2b_sberbank` - Сбербанк Бизнес Онлайн
 * - `tinkoff_bank` - T-Pay
 * - `psb` - ПромсвязьБанк
 * - `installments` - Заплатить по частям
 * - `wechat` - Платеж через WeChat
 * - `sbp` - Платеж через сервис быстрых платежей
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class PaymentMethodType extends AbstractEnum
{
    /** Платеж из кошелька ЮMoney */
    public const YOO_MONEY = 'yoo_money';

    /** Платеж с произвольной банковской карты */
    public const BANK_CARD = 'bank_card';

    /** Платеж СбербанкОнлайн */
    public const SBERBANK = 'sberbank';

    /** Платеж наличными */
    public const CASH = 'cash';

    /** Платеж с баланса мобильного телефона */
    public const MOBILE_BALANCE = 'mobile_balance';

    /** Платеж ApplePay */
    public const APPLE_PAY = 'apple_pay';

    /** Платеж Google Pay */
    public const GOOGLE_PAY = 'google_pay';

    /**
     * Платеж из кошелька Qiwi
     *
     * @deprecated Больше недоступен
     */
    public const QIWI = 'qiwi';

    /**
     * Платеж из кошелька Webmoney
     *
     * @deprecated Больше недоступен
     */
    public const WEBMONEY = 'webmoney';

    /**
     * Платеж через Альфа-Клик
     *
     * @deprecated Больше недоступен
     */
    public const ALFABANK = 'alfabank';

    /** Сбербанк Бизнес Онлайн */
    public const B2B_SBERBANK = 'b2b_sberbank';

    /** Платеж через T-Pay */
    public const TINKOFF_BANK = 'tinkoff_bank';

    /**
     * ПромсвязьБанк
     *
     * @deprecated Больше недоступен
     */
    public const PSB = 'psb';

    /**
     * Заплатить по частям
     *
     * @deprecated Больше недоступен
     */
    public const INSTALLMENTS = 'installments';

    /**
     * Оплата через WeChat.
     *
     * @deprecated Больше недоступен
     */
    public const WECHAT = 'wechat';

    /** Оплата через сервис быстрых платежей */
    public const SBP = 'sbp';

    /** Прием оплаты с использованием Кредита от СберБанка */
    public const SBER_LOAN = 'sber_loan';

    /** Прием платежей по электронному сертификату, привязанному к карте «Мир» */
    public const ELECTRONIC_CERTIFICATE = 'electronic_certificate';

    /** Оплата через сервис «Плати частями» */
    public const SBER_BNPL = 'sber_bnpl';

    /**
     * Для неизвестных методов оплаты
     *
     * @deprecated Не используется для реальных платежей
     */
    public const UNKNOWN = 'unknown';

    protected static array $validValues = [
        self::YOO_MONEY => true,
        self::BANK_CARD => true,
        self::SBERBANK => true,
        self::CASH => true,
        self::MOBILE_BALANCE => true,
        self::APPLE_PAY => false,
        self::GOOGLE_PAY => false,
        self::QIWI => false,
        self::WEBMONEY => false,
        self::ALFABANK => false,
        self::TINKOFF_BANK => true,
        self::INSTALLMENTS => false,
        self::B2B_SBERBANK => true,
        self::PSB => false,
        self::WECHAT => false,
        self::SBP => true,
        self::SBER_LOAN => true,
        self::ELECTRONIC_CERTIFICATE => true,
        self::SBER_BNPL => true,
        self::UNKNOWN => false,
    ];
}
