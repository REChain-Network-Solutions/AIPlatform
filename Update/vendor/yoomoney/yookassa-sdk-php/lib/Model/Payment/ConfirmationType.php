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
 * Класс, представляющий модель ConfirmationType.
 *
 * Тип пользовательского процесса подтверждения платежа.
 *
 * Возможные значения:
 * - `redirect` - Необходимо направить плательщика на страницу партнера
 * - `external` - Для подтверждения платежа пользователю необходимо совершить действия во внешней системе (например, ответить на смс)
 * - `code_verification` - Необходимо получить одноразовый код от плательщика для подтверждения платежа
 * - `embedded` - Необходимо получить токен для checkout.js
 * - `qr` - Необходимо получить QR-код
 * - `mobile_application` - Необходимо совершить действия в мобильном приложении
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class ConfirmationType extends AbstractEnum
{
    /** Необходимо направить плательщика на страницу партнера */
    public const REDIRECT = 'redirect';

    /** Для подтверждения платежа пользователю необходимо совершить действия во внешней системе (например, ответить на смс) */
    public const EXTERNAL = 'external';

    /**
     * Необходимо ждать пока плательщик самостоятельно подтвердит платеж.
     *
     * @deprecated Будет удален в следующих версиях
     */
    public const CODE_VERIFICATION = 'code_verification';

    /** Необходимо получить одноразовый код от плательщика для подтверждения платежа */
    public const EMBEDDED = 'embedded';

    /** Необходимо получить QR-код */
    public const QR = 'qr';

    /** Необходимо совершить действия в мобильном приложении */
    public const MOBILE_APPLICATION = 'mobile_application';

    protected static array $validValues = [
        self::REDIRECT => true,
        self::EXTERNAL => true,
        self::CODE_VERIFICATION => false,
        self::EMBEDDED => true,
        self::QR => true,
        self::MOBILE_APPLICATION => true,
    ];
}
