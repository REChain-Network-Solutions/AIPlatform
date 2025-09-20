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

namespace YooKassa\Model\Invoice\DeliveryMethod;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель DeliveryMethodType.
 *
 * Код способа доставки счета пользователю.
 *
 * Возможные значения:
 * - `self` — Самостоятельно.
 *
 * Подробнее про [жизненный цикл счета](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/invoices/basics)
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class DeliveryMethodType extends AbstractEnum
{
    /** Самостоятельно */
    public const SELF = 'self';
    /**
     * Для неизвестных методов доставки счета пользователю
     *
     * @deprecated Не используется для реальных платежей
     */
    public const UNKNOWN = 'unknown';

    /**
     * Возвращает список доступных значений
     * @return string[]
     */
    protected static array $validValues = [
        self::SELF => true,
        self::UNKNOWN => false,
    ];
}
