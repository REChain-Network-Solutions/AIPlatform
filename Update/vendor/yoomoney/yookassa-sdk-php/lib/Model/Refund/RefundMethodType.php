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

namespace YooKassa\Model\Refund;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель PaymentMethodType.
 *
 * Тип источника средств для проведения платежа.
 *
 * Возможные значения:
 * - `sbp` - Возврат через сервис быстрых платежей
 * - `electronic_certificate` - Возврат платежа по электронному сертификату
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class RefundMethodType extends AbstractEnum
{
    /** Возврат через сервис быстрых платежей */
    public const SBP = 'sbp';
    /** Прием платежей по электронному сертификату, привязанному к карте «Мир» */
    public const ELECTRONIC_CERTIFICATE = 'electronic_certificate';
    /**
     * Для неизвестных методов возврата
     *
     * @deprecated Не используется для реальных возвратов
     */
    public const UNKNOWN = 'unknown';

    protected static array $validValues = [
        self::SBP => true,
        self::ELECTRONIC_CERTIFICATE => true,
        self::UNKNOWN => false,
    ];
}
