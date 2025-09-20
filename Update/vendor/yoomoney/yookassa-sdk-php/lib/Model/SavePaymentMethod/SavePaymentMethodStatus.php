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

namespace YooKassa\Model\SavePaymentMethod;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель SavePaymentMethodStatus.
 *
 * Статус проверки и сохранения способа оплаты. Возможные значения:
 * ~`pending` — ожидает действий от пользователя;
 * ~`active` — способ оплаты сохранен, его можно использовать для автоплатежей или выплат;
 * ~`inactive` — способ оплаты не сохранен: пользователь не подтвердил привязку платежного средства или при сохранении способа оплаты возникла ошибка. Чтобы узнать подробности, обратитесь в техническую поддержку ЮKassa.
 * 
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
*/
class SavePaymentMethodStatus extends AbstractEnum
{
    /** Ожидает действий от пользователя */
    public const PENDING = 'pending';

    /* Способ оплаты сохранен, его можно использовать для автоплатежей или выплат */
    public const ACTIVE = 'active';

    /** Способ оплаты не сохранен: пользователь не подтвердил привязку платежного средства или при сохранении способа оплаты возникла ошибка. Чтобы узнать подробности, обратитесь в техническую поддержку ЮKassa. */
    public const INACTIVE = 'inactive';
    
    /**
     * Возвращает список доступных значений
     * @return string[]
     */
    protected static array $validValues = [
        self::PENDING => true,
        self::ACTIVE => true,
        self::INACTIVE => true
    ];
}

