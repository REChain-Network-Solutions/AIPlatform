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

namespace YooKassa\Model\Receipt;

use YooKassa\Common\AbstractEnum;

/**
 * AgentType - Тип посредника.
 *
 * Тип посредника передается в запросе на создание чека в массиве `items`, в параметре `agent_type`,
 * если вы отправляете данные для формирования чека по сценарию Сначала платеж, потом чек.
 * Параметр `agent_type` нужно передавать, начиная с ФФД 1.1. Убедитесь, что ваша онлайн-касса обновлена до этой версии.
 *
 * Возможные значения:
 * - `banking_payment_agent` - Безналичный расчет
 * - `banking_payment_subagent` - Предоплата (аванс)
 * - `payment_agent` - Постоплата (кредит)
 * - `payment_subagent` - Встречное предоставление
 * - `attorney` - Встречное предоставление
 * - `commissioner` - Встречное предоставление
 * - `agent` - Встречное предоставление
 */
class AgentType extends AbstractEnum
{
    /** Банковский платежный агент */
    public const BANKING_PAYMENT_AGENT = 'banking_payment_agent';

    /** Банковский платежный субагент */
    public const BANKING_PAYMENT_SUBAGENT = 'banking_payment_subagent';

    /** Платежный агент */
    public const PAYMENT_AGENT = 'payment_agent';

    /** Платежный субагент */
    public const PAYMENT_SUBAGENT = 'payment_subagent';

    /** Поверенный */
    public const ATTORNEY = 'attorney';

    /** Комиссионер */
    public const COMMISSIONER = 'commissioner';

    /** Агент */
    public const AGENT = 'agent';

    protected static array $validValues = [
        self::BANKING_PAYMENT_AGENT => true,
        self::BANKING_PAYMENT_SUBAGENT => true,
        self::PAYMENT_AGENT => true,
        self::PAYMENT_SUBAGENT => true,
        self::ATTORNEY => true,
        self::COMMISSIONER => true,
        self::AGENT => true,
    ];
}
