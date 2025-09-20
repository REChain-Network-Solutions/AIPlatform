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

namespace YooKassa\Request\Payouts;

use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\PayoutDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payout\AbstractPayoutDestination;

/**
 * Interface CreatePayoutRequestInterface.
 *
 * @category Interface
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property AmountInterface $amount Сумма создаваемой выплаты
 * @property AbstractPayoutDestination $payoutDestinationData Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token.
 * @property AbstractPayoutDestination $payout_destination_data Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token.
 * @property string $payoutToken Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data
 * @property string $payout_token Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data
 * @property string $payment_method_id Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты
 * @property string $paymentMethodId Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты
 * @property PayoutDealInfo $deal Сделка, в рамках которой нужно провести выплату. Необходимо передавать, если вы проводите Безопасную сделку
 * @property string $description Описание транзакции (не более 128 символов). Например: «Выплата по договору N»
 * @property PayoutSelfEmployedInfo $self_employed Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.
 * @property PayoutSelfEmployedInfo $selfEmployed Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.
 * @property IncomeReceiptData $receipt_data Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.
 * @property IncomeReceiptData $receiptData Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.
 * @property PayoutPersonalData $personal_data Персональные данные получателя выплаты. Необходимо передавать, если вы делаете выплаты с [проверкой получателя](/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП).
 * @property PayoutPersonalData $personalData Персональные данные получателя выплаты. Необходимо передавать, если вы делаете выплаты с [проверкой получателя](/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП).
 * @property Metadata $metadata Метаданные привязанные к выплате
 */
interface CreatePayoutRequestInterface
{
    /**
     * Возвращает сумму выплаты.
     *
     * @return AmountInterface|null Сумма выплаты
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Проверяет наличие суммы в создаваемой выплате.
     *
     * @return bool True если сумма установлена, false если нет
     */
    public function hasAmount(): bool;

    /**
     * Устанавливает сумму выплаты.
     *
     * @param AmountInterface|array $amount Сумма выплаты
     *
     * @return self
     */
    public function setAmount(mixed $amount): self;

    /**
     * Возвращает данные платежного средства, на которое нужно сделать выплату.
     *
     * @return null|AbstractPayoutDestination Данные платежного средства, на которое нужно сделать выплату
     */
    public function getPayoutDestinationData(): ?AbstractPayoutDestination;

    /**
     * Проверяет наличие данных платежного средства, на которое нужно сделать выплату.
     *
     * @return bool True если данные есть, false если нет
     */
    public function hasPayoutDestinationData(): bool;

    /**
     * Устанавливает данные платежного средства, на которое нужно сделать выплату.
     *
     * @param null|AbstractPayoutDestination|array $payout_destination_data Данные платежного средства, на которое нужно сделать выплату
     *
     * @return self
     */
    public function setPayoutDestinationData(mixed $payout_destination_data): self;

    /**
     * Возвращает токенизированные данные для выплаты.
     *
     * @return string|null Токенизированные данные для выплаты
     */
    public function getPayoutToken(): ?string;

    /**
     * Проверяет наличие токенизированных данных для выплаты.
     *
     * @return bool True если токен установлен, false если нет
     */
    public function hasPayoutToken(): bool;

    /**
     * Устанавливает токенизированные данные для выплаты.
     *
     * @param string|null $payout_token Токенизированные данные для выплаты
     *
     * @return self
     */
    public function setPayoutToken(?string $payout_token): self;

    /**
     * Возвращает идентификатор сохраненного способа оплаты.
     *
     * @return null|string Идентификатор сохраненного способа оплаты
     */
    public function getPaymentMethodId(): ?string;

    /**
     * Проверяет наличие идентификатора сохраненного способа оплаты.
     *
     * @return bool True если идентификатора установлен, false если нет
     */
    public function hasPaymentMethodId(): bool;

    /**
     * Устанавливает идентификатор сохраненного способа оплаты.
     *
     * @param null|string $payment_method_id Идентификатор сохраненного способа оплаты
     *
     * @return self
     */
    public function setPaymentMethodId(?string $payment_method_id): self;

    /**
     * Возвращает описание транзакции.
     *
     * @return string|null Описание транзакции
     */
    public function getDescription(): ?string;

    /**
     * Проверяет наличие описания транзакции в создаваемой выплате.
     *
     * @return bool True если описание транзакции установлено, false если нет
     */
    public function hasDescription(): bool;

    /**
     * Устанавливает описание транзакции.
     *
     * @param string|null $description Описание транзакции
     *
     * @return self
     */
    public function setDescription(?string $description): self;

    /**
     * Возвращает сделку, в рамках которой нужно провести выплату.
     *
     * @return null|PayoutDealInfo Сделка, в рамках которой нужно провести выплату
     */
    public function getDeal(): ?PayoutDealInfo;

    /**
     * Проверяет установлена ли сделка, в рамках которой нужно провести выплату.
     *
     * @return bool True если сделка установлена, false если нет
     */
    public function hasDeal(): bool;

    /**
     * Устанавливает сделку, в рамках которой нужно провести выплату.
     *
     * @param null|array|PayoutDealInfo $deal Сделка, в рамках которой нужно провести выплату
     *
     * @return self
     */
    public function setDeal(mixed $deal): self;

    /**
     * Возвращает данные самозанятого, который получит выплату.
     *
     * @return null|PayoutSelfEmployedInfo Данные самозанятого, который получит выплату
     */
    public function getSelfEmployed(): ?PayoutSelfEmployedInfo;

    /**
     * Проверяет наличие данных самозанятого в создаваемой выплате.
     *
     * @return bool True если данные самозанятого есть, false если нет
     */
    public function hasSelfEmployed(): bool;

    /**
     * Устанавливает данные самозанятого, который получит выплату.
     *
     * @param null|array|PayoutSelfEmployedInfo $self_employed Данные самозанятого, который получит выплату
     *
     * @return self
     */
    public function setSelfEmployed(mixed $self_employed): self;

    /**
     * Возвращает данные для формирования чека в сервисе Мой налог.
     *
     * @return null|IncomeReceiptData Данные для формирования чека в сервисе Мой налог
     */
    public function getReceiptData(): ?IncomeReceiptData;

    /**
     * Проверяет наличие данных для формирования чека в сервисе Мой налог.
     *
     * @return bool True если данные для формирования чека есть, false если нет
     */
    public function hasReceiptData(): bool;

    /**
     * Устанавливает данные для формирования чека в сервисе Мой налог..
     *
     * @param null|array|IncomeReceiptData $receipt_data Данные для формирования чека в сервисе Мой налог
     *
     * @return self
     */
    public function setReceiptData(mixed $receipt_data): self;

    /**
     * Возвращает персональные данные получателя выплаты.
     *
     * @return PayoutPersonalData[]|ListObjectInterface Персональные данные получателя выплаты
     */
    public function getPersonalData(): ListObjectInterface;

    /**
     * Проверяет наличие персональных данных в создаваемой выплате.
     *
     * @return bool True если персональные данные есть, false если нет
     */
    public function hasPersonalData(): bool;

    /**
     * Устанавливает персональные данные получателя выплаты.
     *
     * @param null|array|ListObjectInterface $personal_data Персональные данные получателя выплаты
     *
     * @return self
     */
    public function setPersonalData(?array $personal_data = null): self;

    /**
     * Возвращает данные оплаты установленные мерчантом
     *
     * @return Metadata|null Метаданные привязанные к выплате
     */
    public function getMetadata(): ?Metadata;

    /**
     * Проверяет, были ли установлены метаданные заказа.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool;

    /**
     * Устанавливает метаданные, привязанные к выплате.
     *
     * @param null|array|Metadata $metadata Метаданные платежа, устанавливаемые мерчантом
     *
     * @return self
     */
    public function setMetadata(mixed $metadata): self;
}
