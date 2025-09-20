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
use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CancellationDetailsInterface;
use YooKassa\Model\Deal\PaymentDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\Confirmation\AbstractConfirmation;
use YooKassa\Model\Payment\Confirmation\ConfirmationFactory;
use YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod;
use YooKassa\Model\Payment\PaymentMethod\PaymentMethodFactory;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Payment.
 *
 * Объект платежа (Payment) — актуальная информация о платеже.
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
 * @property ListObjectInterface|TransferInterface[] $transfers Данные о распределении платежа между магазинами
 * @property AmountInterface $incomeAmount Сумма платежа, которую получит магазин
 * @property AmountInterface $income_amount Сумма платежа, которую получит магазин
 * @property PaymentDealInfo $deal Данные о сделке, в составе которой проходит платеж
 * @property string $merchantCustomerId Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property string $merchant_customer_id Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property PaymentInvoiceDetails $invoice_details Данные о выставленном счете, в рамках которого проведен платеж
 * @property PaymentInvoiceDetails $invoiceDetails Данные о выставленном счете, в рамках которого проведен платеж
 */
class Payment extends AbstractObject implements PaymentInterface
{
    /**
     * @var string|null Идентификатор платежа
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_ID)]
    #[Assert\Length(min: self::MIN_LENGTH_ID)]
    protected ?string $_id = null;

    /**
     * @var string|null Текущее состояние платежа
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PaymentStatus::class, 'getValidValues'])]
    protected ?string $_status = null;

    /**
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    protected ?AmountInterface $_amount = null;

    /**
     * @var AmountInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    protected ?AmountInterface $_income_amount = null;

    /**
     * @var string|null Описание платежа
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    protected ?string $_description = null;

    /**
     * @var null|RecipientInterface Получатель платежа
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(Recipient::class)]
    protected ?RecipientInterface $_recipient = null;

    /**
     * @var AbstractPaymentMethod|null Способ проведения платежа
     */
    #[Assert\Type(AbstractPaymentMethod::class)]
    protected ?AbstractPaymentMethod $_payment_method = null;

    /**
     * @var DateTime|null Время создания заказа
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_created_at = null;

    /**
     * @var DateTime|null Время подтверждения платежа магазином
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_captured_at = null;

    /**
     * Время, до которого можно бесплатно отменить или подтвердить платеж. В указанное время платеж в статусе
     * `waiting_for_capture` будет автоматически отменен.
     *
     * @var DateTime|null Время, до которого можно бесплатно отменить или подтвердить платеж
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_expires_at = null;

    /**
     * @var AbstractConfirmation|null Способ подтверждения платежа
     */
    #[Assert\Valid]
    #[Assert\Type(AbstractConfirmation::class)]
    protected ?AbstractConfirmation $_confirmation = null;

    /**
     * Признак тестовой операции.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected bool $_test = false;

    /**
     * @var AmountInterface|null Сумма возвращенных средств платежа
     */
    #[Assert\Valid]
    #[Assert\Type(ReceiptItemAmount::class)]
    protected ?AmountInterface $_refunded_amount = null;

    /**
     * @var bool Признак оплаты заказа
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected bool $_paid = false;

    /**
     * @var bool Возможность провести возврат по API
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected bool $_refundable = true;

    /**
     * @var string|null Состояние регистрации фискального чека
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [ReceiptRegistrationStatus::class, 'getValidValues'])]
    protected ?string $_receipt_registration = null;

    /**
     * @var Metadata|null Метаданные платежа указанные мерчантом
     */
    #[Assert\AllType('string')]
    #[Assert\Type(Metadata::class)]
    protected ?Metadata $_metadata = null;

    /**
     * Комментарий к статусу canceled: кто отменил платеж и по какой причине
     * @var CancellationDetailsInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(CancellationDetails::class)]
    protected ?CancellationDetailsInterface $_cancellation_details = null;

    /**
     * Данные об авторизации платежа
     * @var AuthorizationDetailsInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(AuthorizationDetails::class)]
    protected ?AuthorizationDetailsInterface $_authorization_details = null;

    /**
     * @var TransferInterface[]|null Данные о распределении платежа между магазинами
     */
    #[Assert\Valid]
    #[Assert\AllType(Transfer::class)]
    #[Assert\Type(ListObject::class)]
    protected ?ListObject $_transfers = null;

    /**
     * @var PaymentDealInfo|null Данные о сделке, в составе которой проходит платеж. Необходимо передавать, если вы проводите Безопасную сделку
     */
    #[Assert\Valid]
    #[Assert\Type(PaymentDealInfo::class)]
    protected ?PaymentDealInfo $_deal = null;

    /**
     * @var string|null Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов.
     *             Присутствует, если вы хотите запомнить банковскую карту и отобразить ее при повторном платеже в виджете ЮKassa
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_MERCHANT_CUSTOMER_ID)]
    protected ?string $_merchant_customer_id = null;

    /**
     * @var PaymentInvoiceDetails|null
     */
    #[Assert\Type(PaymentInvoiceDetails::class)]
    protected ?PaymentInvoiceDetails $_invoice_details = null;

    /**
     * Возвращает идентификатор платежа.
     *
     * @return string|null Идентификатор платежа
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор платежа.
     *
     * @param string|null $id Идентификатор платежа
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает состояние платежа.
     *
     * @return string|null Текущее состояние платежа
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает статус платежа
     * @param string|null $status Статус платежа
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает сумму.
     *
     * @return AmountInterface|null Сумма платежа
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает сумму платежа.
     *
     * @param AmountInterface|array|null $amount Сумма платежа
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает сумму платежа, которую получит магазин, значение `amount` за вычетом комиссии ЮKassa.
     *
     * @return AmountInterface|null Сумма платежа, которую получит магазин
     */
    public function getIncomeAmount(): ?AmountInterface
    {
        return $this->_income_amount;
    }

    /**
     * Устанавливает сумму платежа, которую получит магазин, значение `amount` за вычетом комиссии ЮKassa
     * @param AmountInterface|array|null $income_amount
     *
     * @return self
     */
    public function setIncomeAmount(mixed $income_amount = null): self
    {
        $this->_income_amount = $this->validatePropertyValue('_income_amount', $income_amount);
        return $this;
    }

    /**
     * Возвращает описание транзакции
     * @return string|null
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает описание транзакции
     *
     * @param string|null $description
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает получателя платежа.
     *
     * @return null|RecipientInterface Получатель платежа или null, если получатель не задан
     */
    public function getRecipient(): ?RecipientInterface
    {
        return $this->_recipient;
    }

    /**
     * Устанавливает получателя платежа.
     *
     * @param RecipientInterface|array|null $recipient Объект с информацией о получателе платежа
     *
     * @return self
     */
    public function setRecipient(mixed $recipient = null): self
    {
        $this->_recipient = $this->validatePropertyValue('_recipient', $recipient);
        return $this;
    }

    /**
     * Возвращает используемый способ проведения платежа.
     *
     * @return AbstractPaymentMethod|null Способ проведения платежа
     */
    public function getPaymentMethod(): ?AbstractPaymentMethod
    {
        return $this->_payment_method;
    }

    /**
     * Устанавливает используемый способ проведения платежа.
     *
     * @param AbstractPaymentMethod|array|null $payment_method Способ проведения платежа
     *
     * @return self
     */
    public function setPaymentMethod(mixed $payment_method): self
    {
        if (is_array($payment_method)) {
            $payment_method = (new PaymentMethodFactory)->factoryFromArray($payment_method);
        }
        $this->_payment_method = $this->validatePropertyValue('_payment_method', $payment_method);
        return $this;
    }

    /**
     * Возвращает время создания заказа.
     *
     * @return DateTime|null Время создания заказа
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает время создания заказа.
     *
     * @param DateTime|string|null $created_at Время создания заказа
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * Возвращает время подтверждения платежа магазином или null, если время не задано.
     *
     * @return null|DateTime Время подтверждения платежа магазином
     */
    public function getCapturedAt(): ?DateTime
    {
        return $this->_captured_at;
    }

    /**
     * Устанавливает время подтверждения платежа магазином
     *
     * @param DateTime|string|null $captured_at Время подтверждения платежа магазином
     *
     * @return self
     */
    public function setCapturedAt(DateTime|string|null $captured_at = null): self
    {
        $this->_captured_at = $this->validatePropertyValue('_captured_at', $captured_at);
        return $this;
    }

    /**
     * Возвращает время до которого можно бесплатно отменить или подтвердить платеж, или null, если оно не задано.
     *
     * @return null|DateTime Время, до которого можно бесплатно отменить или подтвердить платеж
     */
    public function getExpiresAt(): ?DateTime
    {
        return $this->_expires_at;
    }

    /**
     * Устанавливает время до которого можно бесплатно отменить или подтвердить платеж.
     *
     * @param DateTime|string|null $expires_at Время, до которого можно бесплатно отменить или подтвердить платеж
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): self
    {
        $this->_expires_at = $this->validatePropertyValue('_expires_at', $expires_at);
        return $this;
    }

    /**
     * Возвращает способ подтверждения платежа.
     *
     * @return AbstractConfirmation|null Способ подтверждения платежа
     */
    public function getConfirmation(): ?AbstractConfirmation
    {
        return $this->_confirmation;
    }

    /**
     * Устанавливает способ подтверждения платежа.
     *
     * @param AbstractConfirmation|array|null $confirmation Способ подтверждения платежа
     *
     * @return self
     */
    public function setConfirmation(mixed $confirmation = null): self
    {
        if (is_array($confirmation)) {
            $confirmation = (new ConfirmationFactory())->factoryFromArray($confirmation);
        }
        $this->_confirmation = $this->validatePropertyValue('_confirmation', $confirmation);
        return $this;
    }

    /**
     * Возвращает сумму возвращенных средств.
     *
     * @return AmountInterface|null Сумма возвращенных средств платежа
     */
    public function getRefundedAmount(): ?AmountInterface
    {
        return $this->_refunded_amount;
    }

    /**
     * Устанавливает сумму возвращенных средств.
     *
     * @param AmountInterface|array|null $refunded_amount Сумма возвращенных средств платежа
     *
     * @return self
     */
    public function setRefundedAmount(mixed $refunded_amount = null): self
    {
        $this->_refunded_amount = $this->validatePropertyValue('_refunded_amount', $refunded_amount);
        return $this;
    }

    /**
     * Проверяет, был ли уже оплачен заказ.
     *
     * @return bool Признак оплаты заказа, true если заказ оплачен, false если нет
     */
    public function getPaid(): bool
    {
        return $this->_paid;
    }

    /**
     * Устанавливает флаг оплаты заказа.
     *
     * @param bool $paid Признак оплаты заказа
     *
     * @return self
     */
    public function setPaid(bool $paid): self
    {
        $this->_paid = $this->validatePropertyValue('_paid', $paid);
        return $this;
    }

    /**
     * Проверяет возможность провести возврат по API.
     *
     * @return bool Возможность провести возврат по API, true если есть, false если нет
     */
    public function getRefundable(): bool
    {
        return $this->_refundable;
    }

    /**
     * Устанавливает возможность провести возврат по API.
     *
     * @param bool $refundable Возможность провести возврат по API
     *
     * @return self
     */
    public function setRefundable(bool $refundable): self
    {
        $this->_refundable = $this->validatePropertyValue('_refundable', $refundable);
        return $this;
    }

    /**
     * Возвращает состояние регистрации фискального чека.
     *
     * @return string|null Состояние регистрации фискального чека
     */
    public function getReceiptRegistration(): ?string
    {
        return $this->_receipt_registration;
    }

    /**
     * Устанавливает состояние регистрации фискального чека
     *
     * @param string|null $receipt_registration Состояние регистрации фискального чека
     *
     * @return self
     */
    public function setReceiptRegistration(?string $receipt_registration = null): self
    {
        $this->_receipt_registration = $this->validatePropertyValue('_receipt_registration', $receipt_registration);
        return $this;
    }

    /**
     * Возвращает метаданные платежа установленные мерчантом
     *
     * @return Metadata|null Метаданные платежа указанные мерчантом
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает метаданные платежа.
     *
     * @param Metadata|array|null $metadata Метаданные платежа указанные мерчантом
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

    /**
     * Возвращает комментарий к статусу canceled: кто отменил платеж и по какой причине.
     *
     * @return null|CancellationDetailsInterface Комментарий к статусу canceled
     */
    public function getCancellationDetails(): ?CancellationDetailsInterface
    {
        return $this->_cancellation_details;
    }

    /**
     * Устанавливает комментарий к статусу canceled: кто отменил платеж и по какой причине.
     *
     * @param CancellationDetailsInterface|array|null $cancellation_details Комментарий к статусу canceled
     *
     * @return self
     */
    public function setCancellationDetails(mixed $cancellation_details = null): self
    {
        $this->_cancellation_details = $this->validatePropertyValue('_cancellation_details', $cancellation_details);
        return $this;
    }

    /**
     * Возвращает данные об авторизации платежа.
     *
     * @return null|AuthorizationDetailsInterface Данные об авторизации платежа
     */
    public function getAuthorizationDetails(): ?AuthorizationDetailsInterface
    {
        return $this->_authorization_details;
    }

    /**
     * Устанавливает данные об авторизации платежа.
     *
     * @param AuthorizationDetailsInterface|array|null $authorization_details Данные об авторизации платежа
     *
     * @return self
     */
    public function setAuthorizationDetails(mixed $authorization_details = null): self
    {
        $this->_authorization_details = $this->validatePropertyValue('_authorization_details', $authorization_details);
        return $this;
    }

    /**
     * Возвращает массив распределения денег между магазинами.
     *
     * @return TransferInterface[]|ListObjectInterface Массив распределения денег
     */
    public function getTransfers(): ListObjectInterface
    {
        if ($this->_transfers === null) {
            $this->_transfers = new ListObject(Transfer::class);
        }
        return $this->_transfers;
    }

    /**
     * Устанавливает массив распределения денег между магазинами.
     *
     * @param ListObjectInterface|array|null $transfers Массив распределения денег
     *
     * @return self
     */
    public function setTransfers(mixed $transfers = null): self
    {
        $this->_transfers = $this->validatePropertyValue('_transfers', $transfers);
        return $this;
    }

    /**
     * Возвращает признак тестовой операции.
     *
     * @return bool Признак тестовой операции
     */
    public function getTest(): bool
    {
        return $this->_test;
    }

    /**
     * Устанавливает признак тестовой операции.
     *
     * @param bool $test Признак тестовой операции
     *
     * @return self
     */
    public function setTest(mixed $test = null): self
    {
        $this->_test = $this->validatePropertyValue('_test', $test);
        return $this;
    }

    /**
     * Возвращает данные о сделке, в составе которой проходит платеж.
     *
     * @return PaymentDealInfo|null Данные о сделке, в составе которой проходит платеж
     */
    public function getDeal(): ?PaymentDealInfo
    {
        return $this->_deal;
    }

    /**
     * Устанавливает данные о сделке, в составе которой проходит платеж.
     *
     * @param null|array|PaymentDealInfo $deal Данные о сделке, в составе которой проходит платеж
     *
     * @return self
     */
    public function setDeal(mixed $deal = null): self
    {
        $this->_deal = $this->validatePropertyValue('_deal', $deal);
        return $this;
    }

    /**
     * Возвращает идентификатор покупателя в вашей системе.
     *
     * @return string|null Идентификатор покупателя в вашей системе
     */
    public function getMerchantCustomerId(): ?string
    {
        return $this->_merchant_customer_id;
    }

    /**
     * Устанавливает идентификатор покупателя в вашей системе.
     *
     * @param string|null $merchant_customer_id Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов
     *
     * @return self
     */
    public function setMerchantCustomerId(?string $merchant_customer_id = null): self
    {
        $this->_merchant_customer_id = $this->validatePropertyValue('_merchant_customer_id', $merchant_customer_id);
        return $this;
    }

    /**
     * Возвращает invoice_details.
     *
     * @return PaymentInvoiceDetails|null
     */
    public function getInvoiceDetails(): ?PaymentInvoiceDetails
    {
        return $this->_invoice_details;
    }

    /**
     * Устанавливает invoice_details.
     *
     * @param PaymentInvoiceDetails|array|null $invoice_details
     *
     * @return self
     */
    public function setInvoiceDetails(mixed $invoice_details = null): self
    {
        $this->_invoice_details = $this->validatePropertyValue('_invoice_details', $invoice_details);
        return $this;
    }

}
