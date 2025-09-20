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
use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CancellationDetailsInterface;
use YooKassa\Model\Deal\RefundDealInfo;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\ReceiptRegistrationStatus;
use YooKassa\Model\Refund\RefundMethod\AbstractRefundMethod;
use YooKassa\Model\Refund\RefundMethod\RefundMethodFactory;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Refund.
 *
 * Объект возврата (Refund) — актуальная информация о возврате платежа.
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
 * @property ListObjectInterface|SourceInterface[] $sources Данные о том, с какого магазина и какую сумму нужно удержать для проведения возврата
 * @property RefundDealInfo $deal Данные о сделке, в составе которой проходит возврат
 * @property AbstractRefundMethod|null $refundMethod Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа
 * @property AbstractRefundMethod|null $refund_method Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа
 * @property RefundAuthorizationDetails|null $refund_authorization_details Данные об авторизации возврата. Присутствуют только для возвратов платежей, совершенных этими способами оплаты: банковская карта, Mir Pay
 * @property RefundAuthorizationDetails|null $refundAuthorizationDetails Данные об авторизации возврата. Присутствуют только для возвратов платежей, совершенных этими способами оплаты: банковская карта, Mir Pay
 */
class Refund extends AbstractObject implements RefundInterface
{
    /** Максимальная длина строки описания возврата */
    public const MAX_LENGTH_DESCRIPTION = 250;

    /**
     * @var string|null Идентификатор возврата платежа
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 36)]
    #[Assert\Length(min: 36)]
    protected ?string $_id = null;

    /**
     * @var string|null Идентификатор платежа
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 36)]
    #[Assert\Length(min: 36)]
    protected ?string $_payment_id = null;

    /**
     * @var string|null Статус возврата
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [RefundStatus::class, 'getValidValues'])]
    protected ?string $_status = null;

    /**
     * @var CancellationDetailsInterface|null Комментарий к статусу `canceled`
     */
    #[Assert\Valid]
    #[Assert\Type(RefundCancellationDetails::class)]
    protected ?CancellationDetailsInterface $_cancellation_details = null;

    /**
     * @var string|null Статус регистрации чека
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [ReceiptRegistrationStatus::class, 'getValidValues'])]
    protected ?string $_receipt_registration = null;

    /**
     * @var DateTime|null Время создания возврата
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_created_at = null;

    /**
     * @var AmountInterface|null Сумма возврата
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    protected ?AmountInterface $_amount = null;

    /**
     * @var string|null Комментарий, основание для возврата средств покупателю
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    protected ?string $_description = null;

    /**
     * @var SourceInterface[]|ListObjectInterface|null Данные о распределении денег — сколько и в какой магазин нужно перевести
     */
    #[Assert\Valid]
    #[Assert\AllType(Source::class)]
    #[Assert\Type(ListObject::class)]
    protected ?ListObject $_sources = null;

    /**
     * @var null|RefundDealInfo Данные о сделке, в составе которой проходит возврат
     */
    #[Assert\Valid]
    #[Assert\Type(RefundDealInfo::class)]
    protected ?RefundDealInfo $_deal = null;

    /**
     * @var AbstractRefundMethod|null Детали возврата. Зависят от способа оплаты, который использовался при проведении платежа.
     */
    #[Assert\Type(AbstractRefundMethod::class)]
    protected ?AbstractRefundMethod $_refund_method = null;

    /**
     * @var RefundAuthorizationDetails|null Данные об авторизации возврата
     */
    #[Assert\Valid]
    #[Assert\Type(RefundAuthorizationDetails::class)]
    protected ?RefundAuthorizationDetails $_refund_authorization_details = null;

    /**
     * Возвращает идентификатор возврата платежа.
     *
     * @return string|null Идентификатор возврата
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор возврата.
     *
     * @param string|null $id Идентификатор возврата
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает идентификатор платежа.
     *
     * @return string|null Идентификатор платежа
     */
    public function getPaymentId(): ?string
    {
        return $this->_payment_id;
    }

    /**
     * Устанавливает идентификатор платежа.
     *
     * @param string|null $payment_id Идентификатор платежа
     *
     * @return self
     */
    public function setPaymentId(?string $payment_id = null): self
    {
        $this->_payment_id = $this->validatePropertyValue('_payment_id', $payment_id);
        return $this;
    }

    /**
     * Возвращает статус текущего возврата.
     *
     * @return string|null Статус возврата
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает статус возврата платежа.
     *
     * @param string|null $status Статус возврата платежа
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает cancellation_details.
     *
     * @return CancellationDetailsInterface|null
     */
    public function getCancellationDetails(): ?CancellationDetailsInterface
    {
        return $this->_cancellation_details;
    }

    /**
     * Устанавливает cancellation_details.
     *
     * @param CancellationDetailsInterface|array|null $cancellation_details
     *
     * @return self
     */
    public function setCancellationDetails(mixed $cancellation_details = null): self
    {
        $this->_cancellation_details = $this->validatePropertyValue('_cancellation_details', $cancellation_details);
        return $this;
    }

    /**
     * Возвращает статус регистрации чека.
     *
     * @return string|null Статус регистрации чека
     */
    public function getReceiptRegistration(): ?string
    {
        return $this->_receipt_registration;
    }

    /**
     * Устанавливает статус регистрации чека.
     *
     * @param string|null $receipt_registration Статус регистрации чека
     *
     * @return self
     */
    public function setReceiptRegistration(?string $receipt_registration = null): self
    {
        $this->_receipt_registration = $this->validatePropertyValue('_receipt_registration', $receipt_registration);
        return $this;
    }

    /**
     * Возвращает дату создания возврата.
     *
     * @return DateTime|null Время создания возврата
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает время создания возврата.
     *
     * @param DateTime|string|null $created_at Время создания возврата
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * Возвращает сумму возврата.
     *
     * @return AmountInterface|null Сумма возврата
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает сумму возврата.
     *
     * @param AmountInterface|array|null $amount Сумма возврата
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает комментарий к возврату.
     *
     * @return string|null Комментарий, основание для возврата средств покупателю
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает комментарий к возврату.
     *
     * @param string|null $description Комментарий, основание для возврата средств покупателю
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает информацию о распределении денег — сколько и в какой магазин нужно перевести.
     *
     * @return SourceInterface[]|ListObjectInterface
     */
    public function getSources(): ListObjectInterface
    {
        if ($this->_sources === null) {
            $this->_sources = new ListObject(Source::class);
        }
        return $this->_sources;
    }

    /**
     * Устанавливает sources (массив распределения денег между магазинами).
     *
     * @param ListObjectInterface|array|null $sources
     *
     * @return self
     */
    public function setSources(mixed $sources = null): self
    {
        $this->_sources = $this->validatePropertyValue('_sources', $sources);
        return $this;
    }

    /**
     * Возвращает данные о сделке, в составе которой проходит возврат
     *
     * @return null|RefundDealInfo Данные о сделке, в составе которой проходит возврат
     */
    public function getDeal(): ?RefundDealInfo
    {
        return $this->_deal;
    }

    /**
     * Устанавливает данные о сделке, в составе которой проходит возврат.
     *
     * @param RefundDealInfo|array|null $deal Данные о сделке, в составе которой проходит возврат
     *
     * @return self
     */
    public function setDeal(mixed $deal = null): self
    {
        $this->_deal = $this->validatePropertyValue('_deal', $deal);
        return $this;
    }

    /**
     * Возвращает метод возврата.
     *
     * @return AbstractRefundMethod|null
     */
    public function getRefundMethod(): ?AbstractRefundMethod
    {
        return $this->_refund_method;
    }

    /**
     * Устанавливает метод возврата.
     *
     * @param AbstractRefundMethod|array|null $refund_method
     *
     * @return self
     */
    public function setRefundMethod(mixed $refund_method = null): self
    {
        if (is_array($refund_method)) {
            $refund_method = (new RefundMethodFactory)->factoryFromArray($refund_method);
        }
        $this->_refund_method = $this->validatePropertyValue('_refund_method', $refund_method);
        return $this;
    }

    /**
     * Возвращает refund_authorization_details.
     *
     * @return RefundAuthorizationDetails|null
     */
    public function getRefundAuthorizationDetails(): ?RefundAuthorizationDetails
    {
        return $this->_refund_authorization_details;
    }

    /**
     * Устанавливает refund_authorization_details.
     *
     * @param RefundAuthorizationDetails|array|null $refund_authorization_details Данные об авторизации возврата
     *
     * @return self
     */
    public function setRefundAuthorizationDetails(mixed $refund_authorization_details = null): self
    {
        $this->_refund_authorization_details = $this->validatePropertyValue('_refund_authorization_details', $refund_authorization_details);
        return $this;
    }
}
