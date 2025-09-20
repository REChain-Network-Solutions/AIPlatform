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

namespace YooKassa\Model\Notification;

use YooKassa\Common\AbstractEnum;

/**
 * NotificationEventType - Тип уведомления.
 *
 * Возможные значения:
 * - `payment.waiting_for_capture` - Успешно оплачен покупателем, ожидает подтверждения магазином (capture или aviso)
 * - `payment.succeeded` - Успешно оплачен и подтвержден магазином
 * - `payment.canceled` - Неуспех оплаты или отменен магазином
 * - `refund.succeeded` - Успешный возврат
 * - `deal.closed` - Сделка перешла в статус closed
 * - `payout.canceled` - Выплата перешла в статус canceled
 * - `payout.succeeded` - Выплата перешла в статус succeeded
 */
class NotificationEventType extends AbstractEnum
{
    /** Успешно оплачен покупателем, ожидает подтверждения магазином */
    public const PAYMENT_WAITING_FOR_CAPTURE = 'payment.waiting_for_capture';

    /** Успешно оплачен и подтвержден магазином */
    public const PAYMENT_SUCCEEDED = 'payment.succeeded';

    /** Неуспех оплаты или отменен магазином */
    public const PAYMENT_CANCELED = 'payment.canceled';

    /** Успешный возврат */
    public const REFUND_SUCCEEDED = 'refund.succeeded';

    /** Сделка перешла в статус closed */
    public const DEAL_CLOSED = 'deal.closed';

    /** Выплата перешла в статус canceled */
    public const PAYOUT_CANCELED = 'payout.canceled';

    /** Выплата перешла в статус succeeded */
    public const PAYOUT_SUCCEEDED = 'payout.succeeded';

    protected static array $validValues = [
        self::PAYMENT_WAITING_FOR_CAPTURE => true,
        self::PAYMENT_SUCCEEDED => true,
        self::PAYMENT_CANCELED => true,
        self::REFUND_SUCCEEDED => true,
        self::DEAL_CLOSED => true,
        self::PAYOUT_CANCELED => true,
        self::PAYOUT_SUCCEEDED => true,
    ];
}
