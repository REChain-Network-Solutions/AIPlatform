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

namespace YooKassa\Model\Invoice;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель InvoiceCancellationDetailsReasonCode.
 *
 * Возможные причины отмены счета.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class InvoiceCancellationDetailsReasonCode extends AbstractEnum
{
    /**
     * [Счет отменен вручную](https://yookassa.ru/docs/support/merchant/invoices-to-clients/invoicing#invoicing__cancel) из личного кабинета ЮKassa.
     */
    public const INVOICE_CANCELED = 'invoice_canceled';

    /** Истек срок действия счета, который вы установили в запросе на создание счета в параметре `expires_at`, и по счету нет ни одного успешного платежа */
    public const INVOICE_EXPIRED = 'invoice_expired';

    /** Причина не детализирована, поэтому пользователю следует обратиться к инициатору отмены счета за уточнением подробностей */
    public const GENERAL_DECLINE = 'general_decline';

    /** [Платеж отменен по API](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#cancel) при оплате в две стадии */
    public const PAYMENT_CANCELED = 'payment_canceled';

    /**
     * [Истек срок списания оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#hold) для платежа в две стадии
     */
    public const PAYMENT_EXPIRED_ON_CAPTURE = 'payment_expired_on_capture';

    protected static array $validValues = [
        self::INVOICE_CANCELED => true,
        self::INVOICE_EXPIRED => true,
        self::GENERAL_DECLINE => true,
        self::PAYMENT_CANCELED => true,
        self::PAYMENT_EXPIRED_ON_CAPTURE => true,
    ];
}
