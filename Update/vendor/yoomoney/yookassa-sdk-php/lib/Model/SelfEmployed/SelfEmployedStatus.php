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

namespace YooKassa\Model\SelfEmployed;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель SelfEmployedStatus.
 *
 * Статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков.
 *
 * Возможные значения:
 * - `pending` — ЮMoney запросили права на регистрацию чеков, но самозанятый еще не ответил на заявку;
 * - `confirmed` — самозанятый выдал права ЮMoney; вы можете делать выплаты;
 * - `canceled` — самозанятый отклонил заявку или отозвал ранее выданные права;
 * - `unregistered` — самозанятый с таким ИНН не зарегистрирован в сервисе Мой налог, или пользователь потерял статус самозанятого.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class SelfEmployedStatus extends AbstractEnum
{
    /** ЮMoney запросили права на регистрацию чеков, но самозанятый еще не ответил на заявку */
    public const PENDING = 'pending';

    /** Самозанятый выдал права ЮMoney, вы можете делать выплаты */
    public const CONFIRMED = 'confirmed';

    /** Самозанятый отклонил заявку или отозвал ранее выданные права */
    public const CANCELED = 'canceled';

    /** Самозанятый с таким ИНН не зарегистрирован в сервисе Мой налог, или пользователь потерял статус самозанятого */
    public const UNREGISTERED = 'unregistered';

    /**
     * Возвращает список доступных значений
     * @return string[]
     */
    protected static array $validValues = [
        self::PENDING => true,
        self::CONFIRMED => true,
        self::CANCELED => true,
        self::UNREGISTERED => true,
    ];
}
