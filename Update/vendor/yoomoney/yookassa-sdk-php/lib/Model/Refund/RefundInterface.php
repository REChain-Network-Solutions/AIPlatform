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

namespace YooKassa\Model\Refund;

use DateTime;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\RefundDealInfo;

/**
 * Interface RefundInterface.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор возврата платежа
 * @property string $paymentId Идентификатор платежа
 * @property string $payment_id Идентификатор платежа
 * @property string $status Статус возврата
 * @property RefundCancellationDetails $cancellationDetails Комментарий к статусу `canceled`
 * @property RefundCancellationDetails $cancellation_details Комментарий к статусу `canceled`
 * @property DateTime $createdAt Время создания возврата
 * @property DateTime $created_at Время создания возврата
 * @property AmountInterface $amount Сумма возврата
 * @property string $receiptRegistration Статус регистрации чека
 * @property string $receipt_registration Статус регистрации чека
 * @property string $description Комментарий, основание для возврата средств покупателю
 * @property SourceInterface[] $sources Данные о том, с какого магазина и какую сумму нужно удержать для проведения возврата
 * @property RefundDealInfo $deal Данные о сделке, в составе которой проходит возврат
 */
interface RefundInterface
{
    /**
     * Возвращает идентификатор возврата платежа.
     *
     * @return string|null Идентификатор возврата
     */
    public function getId(): ?string;

    /**
     * Возвращает идентификатор платежа.
     *
     * @return string|null Идентификатор платежа
     */
    public function getPaymentId(): ?string;

    /**
     * Возвращает статус текущего возврата.
     *
     * @return string|null Статус возврата
     */
    public function getStatus(): ?string;

    /**
     * Возвращает дату создания возврата.
     *
     * @return DateTime|null Время создания возврата
     */
    public function getCreatedAt(): ?DateTime;

    /**
     * Возвращает сумму возврата.
     *
     * @return AmountInterface|null Сумма возврата
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Возвращает статус регистрации чека.
     *
     * @return string|null Статус регистрации чека
     */
    public function getReceiptRegistration(): ?string;

    /**
     * Возвращает комментарий к возврату.
     *
     * @return string|null Комментарий, основание для возврата средств покупателю
     */
    public function getDescription(): ?string;

    /**
     * Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести.
     *
     * @return SourceInterface[]|ListObjectInterface
     */
    public function getSources(): ListObjectInterface;

    /**
     * Возвращает сделку, в рамках которой нужно провести возврат.
     *
     * @return null|RefundDealInfo Сделка, в рамках которой нужно провести возврат
     */
    public function getDeal(): ?RefundDealInfo;
}
