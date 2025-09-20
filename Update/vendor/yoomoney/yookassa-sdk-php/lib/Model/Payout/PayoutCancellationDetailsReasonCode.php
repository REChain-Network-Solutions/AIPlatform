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

namespace YooKassa\Model\Payout;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель PayoutCancellationDetailsReasonCode.
 *
 * Возможные причины отмены выплаты. Возможные значения:
 * - `insufficient_funds` - На балансе выплат не хватает денег для проведения выплаты
 * - `fraud_suspected` - Выплата заблокирована из-за подозрения в мошенничестве
 * - `one_time_limit_exceeded` - Превышен лимит на разовое зачисление
 * - `periodic_limit_exceeded` - Превышен лимит выплат за период времени
 * - `rejected_by_payee` - Эмитент отклонил выплату по неизвестным причинам
 * - `general_decline` - Причина не детализирована
 * - `issuer_unavailable` - Эквайер недоступен
 * - `recipient_not_found` - Для выплат через СБП: получатель не найден
 * - `recipient_check_failed` - Только для выплат с проверкой получателя. Получатель выплаты не прошел проверку
 * - `identification_required` - Кошелек ЮMoney не идентифицирован. Пополнение анонимного кошелька запрещено
 * - `self_employed_annual_limit_exceeded` - Превышен лимит на годовой доход самозанятого.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class PayoutCancellationDetailsReasonCode extends AbstractEnum
{
    /** На балансе выплат не хватает денег для проведения выплаты. [Пополните баланс](https://yookassa.ru/docs/support/payouts#balance) и повторите запрос с новым ключом идемпотентности. */
    public const INSUFFICIENT_FUNDS = 'insufficient_funds';

    /** Выплата заблокирована из-за подозрения в мошенничестве. Следует обратиться к [инициатору отмены выплаты](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#declined-payouts-cancellation-details-party) за уточнением подробностей или выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту) */
    public const FRAUD_SUSPECTED = 'fraud_suspected';

    /** Превышен [лимит на разовое зачисление](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#specifics). Можно уменьшить размер выплаты, разбить сумму и сделать несколько выплат, выбрать другой способ получения выплат или другое платежное средство (например, другую банковскую карту) */
    public const ONE_TIME_LIMIT_EXCEEDED = 'one_time_limit_exceeded';

    /** Превышен [лимит выплат за период времени](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#specifics) (сутки, месяц). Следует выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту). */
    public const PERIODIC_LIMIT_EXCEEDED = 'periodic_limit_exceeded';

    /** Эмитент отклонил выплату по неизвестным причинам. Пользователю следует обратиться к эмитенту за уточнением подробностей или выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту). */
    public const REJECTED_BY_PAYEE = 'rejected_by_payee';

    /** Причина не детализирована. Пользователю следует обратиться к [инициатору отмены выплаты](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#declined-payouts-cancellation-details-party) за уточнением подробностей */
    public const GENERAL_DECLINE = 'general_decline';

    /** Эквайер недоступен. Следует выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту) или повторить запрос позже с новым ключом идемпотентности. */
    public const ISSUER_UNAVAILABLE = 'issuer_unavailable';

    /** Для [выплат через СБП](https://yookassa.ru/developers/payouts/making-payouts/sbp): получатель не найден — в выбранном банке или платежном сервисе не найден счет, к которому привязан указанный номер телефона. Следует повторить запрос с новыми данными и новым ключом идемпотентности или выбрать другой способ получения выплаты. */
    public const RECIPIENT_NOT_FOUND = 'recipient_not_found';

    /** Только для выплат с [проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check). Получатель выплаты не прошел проверку: имя получателя не совпало с именем владельца счета, на который необходимо перевести деньги. [Что делать в этом случае](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check#process-results-canceled-recipient-check-failed) */
    public const RECIPIENT_CHECK_FAILED = 'recipient_check_failed';

    /** Кошелек ЮMoney не идентифицирован. Пополнение анонимного кошелька запрещено. Пользователю необходимо [идентифицировать кошелек](https://yoomoney.ru/page?id=536136) */
    public const IDENTIFICATION_REQUIRED = 'identification_required';

    /** Только для [выплат самозанятым](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Превышен лимит на годовой доход самозанятого. Порекомендуйте получателю выплаты проверить свой доход. */
    public const SELF_EMPLOYED_ANNUAL_LIMIT_EXCEEDED = 'self_employed_annual_limit_exceeded';

    /**
     * Возвращает список доступных значений
     * @return string[]
     */
    protected static array $validValues = [
        self::INSUFFICIENT_FUNDS => true,
        self::FRAUD_SUSPECTED => true,
        self::ONE_TIME_LIMIT_EXCEEDED => true,
        self::PERIODIC_LIMIT_EXCEEDED => true,
        self::GENERAL_DECLINE => true,
        self::ISSUER_UNAVAILABLE => true,
        self::RECIPIENT_NOT_FOUND => true,
        self::RECIPIENT_CHECK_FAILED => true,
        self::REJECTED_BY_PAYEE => true,
        self::IDENTIFICATION_REQUIRED => true,
        self::SELF_EMPLOYED_ANNUAL_LIMIT_EXCEEDED => true,
    ];
}
