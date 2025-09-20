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

use DateTime;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CancellationDetailsInterface;
use YooKassa\Model\Deal\PayoutDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod;

/**
 * Interface DealInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $id Идентификатор выплаты
 * @property AmountInterface $amount Сумма выплаты
 * @property string $status Текущее состояние выплаты
 * @property AbstractPaymentMethod $payoutDestination Способ проведения выплаты
 * @property AbstractPaymentMethod $payout_destination Способ проведения выплаты
 * @property string $description Описание транзакции
 * @property DateTime $createdAt Время создания заказа
 * @property DateTime $created_at Время создания заказа
 * @property DateTime $succeededAt Время успешного проведения выплаты
 * @property DateTime $succeeded_at Время успешного проведения выплаты
 * @property PayoutDealInfo $deal Сделка, в рамках которой нужно провести выплату
 * @property CancellationDetailsInterface $cancellationDetails Комментарий к отмене выплаты
 * @property CancellationDetailsInterface $cancellation_details Комментарий к отмене выплаты
 * @property Metadata $metadata Метаданные платежа указанные мерчантом
 * @property bool $test Признак тестовой операции
 */
interface PayoutInterface
{
    /**
     * Возвращает Id сделки.
     *
     * @return string|null Id сделки
     */
    public function getId(): ?string;

    /**
     * Возвращает баланс сделки.
     *
     * @return AmountInterface|null Баланс сделки
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Возвращает статус сделки.
     *
     * @return string|null Статус сделки
     */
    public function getStatus(): ?string;

    /**
     * Возвращает платежное средство, на которое ЮKassa зачисляет выплату.
     *
     * @return AbstractPayoutDestination|null Платежное средство, на которое ЮKassa зачисляет выплату.
     */
    public function getPayoutDestination(): ?AbstractPayoutDestination;

    /**
     * Возвращает описание транзакции (не более 128 символов).
     *
     * @return string|null Описание транзакции
     */
    public function getDescription(): ?string;

    /**
     * Возвращает время создания выплаты.
     *
     * @return DateTime|null Время создания сделки
     */
    public function getCreatedAt(): ?DateTime;

    /**
     * Возвращает время успешного проведения выплаты.
     *
     * @return DateTime|null Время создания выплаты
     */
    public function getSucceededAt(): ?DateTime;

    /**
     * Возвращает сделку, в рамках которой нужно провести выплату.
     *
     * @return PayoutDealInfo|null Сделка, в рамках которой нужно провести выплату
     */
    public function getDeal(): ?PayoutDealInfo;

    /**
     * Возвращает комментарий к статусу canceled: кто отменил выплату и по какой причине.
     *
     * @return null|PayoutCancellationDetails Комментарий к статусу canceled: кто отменил выплату и по какой причине
     */
    public function getCancellationDetails(): ?PayoutCancellationDetails;

    /**
     * Возвращает дополнительные данные сделки.
     *
     * @return Metadata|null Дополнительные данные сделки
     */
    public function getMetadata(): ?Metadata;

    /**
     * Возвращает признак тестовой операции.
     *
     * @return bool|null Признак тестовой операции
     */
    public function getTest(): ?bool;
}
