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

use DateTime;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CancellationDetailsInterface;
use YooKassa\Model\Deal\PaymentDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Confirmation\AbstractConfirmation;
use YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod;

/**
 * Interface PaymentInterface.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор платежа
 * @property string $status Текущее состояние платежа
 * @property RecipientInterface $recipient Получатель платежа
 * @property AmountInterface $amount Сумма заказа
 * @property string $description Описание транзакции
 * @property AbstractPaymentMethod $paymentMethod Способ проведения платежа
 * @property AbstractPaymentMethod $payment_method Способ проведения платежа
 * @property DateTime $createdAt Время создания заказа
 * @property DateTime $created_at Время создания заказа
 * @property DateTime $capturedAt Время подтверждения платежа магазином
 * @property DateTime $captured_at Время подтверждения платежа магазином
 * @property DateTime $expiresAt Время, до которого можно бесплатно отменить или подтвердить платеж
 * @property DateTime $expires_at Время, до которого можно бесплатно отменить или подтвердить платеж
 * @property AbstractConfirmation $confirmation Способ подтверждения платежа
 * @property AmountInterface $refundedAmount Сумма возвращенных средств платежа
 * @property AmountInterface $refunded_amount Сумма возвращенных средств платежа
 * @property bool $paid Признак оплаты заказа
 * @property bool $refundable Возможность провести возврат по API
 * @property string $receiptRegistration Состояние регистрации фискального чека
 * @property string $receipt_registration Состояние регистрации фискального чека
 * @property Metadata $metadata Метаданные платежа указанные мерчантом
 * @property bool $test Признак тестовой операции
 * @property CancellationDetailsInterface $cancellationDetails Комментарий к отмене платежа
 * @property CancellationDetailsInterface $cancellation_details Комментарий к отмене платежа
 * @property AuthorizationDetailsInterface $authorizationDetails Данные об авторизации платежа
 * @property AuthorizationDetailsInterface $authorization_details Данные об авторизации платежа
 * @property TransferInterface[] $transfers Данные о распределении платежа между магазинами
 * @property AmountInterface $incomeAmount Сумма платежа, которую получит магазин
 * @property AmountInterface $income_amount Сумма платежа, которую получит магазин
 * @property PaymentDealInfo $deal Данные о сделке, в составе которой проходит платеж
 * @property string $merchantCustomerId Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property string $merchant_customer_id Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property PaymentInvoiceDetails $invoice_details Данные о выставленном счете, в рамках которого проведен платеж
 * @property PaymentInvoiceDetails $invoiceDetails Данные о выставленном счете, в рамках которого проведен платеж
 */
interface PaymentInterface
{
    /** @var int Минимальная длина идентификатора */
    public const MIN_LENGTH_ID = 36;

    /** @var int Максимальная длина идентификатора */
    public const MAX_LENGTH_ID = 36;

    /** Максимальная длина строки описания платежа */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /** Максимальная длина строки идентификатора покупателя в вашей системе */
    public const MAX_LENGTH_MERCHANT_CUSTOMER_ID = 200;

    /**
     * Возвращает идентификатор платежа.
     *
     * @return string|null Идентификатор платежа
     */
    public function getId(): ?string;

    /**
     * Возвращает состояние платежа.
     *
     * @return string|null Текущее состояние платежа
     */
    public function getStatus(): ?string;

    /**
     * Возвращает получателя платежа.
     *
     * @return null|RecipientInterface Получатель платежа или null, если получатель не задан
     */
    public function getRecipient(): ?RecipientInterface;

    /**
     * Возвращает сумму.
     *
     * @return AmountInterface|null Сумма платежа
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Возвращает используемый способ проведения платежа.
     *
     * @return AbstractPaymentMethod|null Способ проведения платежа
     */
    public function getPaymentMethod(): ?AbstractPaymentMethod;

    /**
     * Возвращает время создания заказа.
     *
     * @return DateTime|null Время создания заказа
     */
    public function getCreatedAt(): ?DateTime;

    /**
     * Возвращает время подтверждения платежа магазином или null, если время не задано.
     *
     * @return null|DateTime Время подтверждения платежа магазином
     */
    public function getCapturedAt(): ?DateTime;

    /**
     * Возвращает способ подтверждения платежа.
     *
     * @return AbstractConfirmation|null Способ подтверждения платежа
     */
    public function getConfirmation(): ?AbstractConfirmation;

    /**
     * Возвращает сумму возвращенных средств.
     *
     * @return AmountInterface|null Сумма возвращенных средств платежа
     */
    public function getRefundedAmount(): ?AmountInterface;

    /**
     * Проверяет, был ли уже оплачен заказ.
     *
     * @return bool Признак оплаты заказа, true если заказ оплачен, false если нет
     */
    public function getPaid(): bool;

    /**
     * Возможность провести возврат по API.
     *
     * @return bool Возможность провести возврат по API
     */
    public function getRefundable(): bool;

    /**
     * Возвращает состояние регистрации фискального чека.
     *
     * @return string|null Состояние регистрации фискального чека
     */
    public function getReceiptRegistration(): ?string;

    /**
     * Возвращает метаданные платежа установленные мерчантом
     *
     * @return Metadata|null Метаданные платежа указанные мерчантом
     */
    public function getMetadata(): ?Metadata;

    /**
     * Возвращает время до которого можно бесплатно отменить или подтвердить платеж, или null, если оно не задано.
     *
     * @return null|DateTime Время, до которого можно бесплатно отменить или подтвердить платеж
     */
    public function getExpiresAt(): ?DateTime;

    /**
     * Возвращает комментарий к статусу canceled: кто отменил платеж и по какой причине.
     *
     * @return null|CancellationDetailsInterface Комментарий к статусу canceled
     */
    public function getCancellationDetails(): ?CancellationDetailsInterface;

    /**
     * Возвращает данные об авторизации платежа.
     *
     * @return null|AuthorizationDetailsInterface Данные об авторизации платежа
     */
    public function getAuthorizationDetails(): ?AuthorizationDetailsInterface;

    /**
     * Возвращает данные о распределении платежа между магазинами.
     *
     * @return TransferInterface[]|ListObjectInterface
     */
    public function getTransfers(): ListObjectInterface;

    /**
     * Возвращает сумму перечисляемая магазину за вычетом комиссий платежной системы.(только для успешных платежей).
     */
    public function getIncomeAmount(): ?AmountInterface;

    /**
     * Возвращает сделку, в рамках которой нужно провести платеж.
     *
     * @return PaymentDealInfo|null Сделка, в рамках которой нужно провести платеж
     */
    public function getDeal(): ?PaymentDealInfo;

    /**
     * Возвращает invoice_details.
     *
     * @return PaymentInvoiceDetails|null
     */
    public function getInvoiceDetails(): ?PaymentInvoiceDetails;
}
