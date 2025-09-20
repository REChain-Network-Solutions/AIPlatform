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
use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\CancellationDetailsInterface;
use YooKassa\Model\Deal\PayoutDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Payout.
 *
 * Объект выплаты (Payout) — актуальная информация о выплате.
 *
 * @category Class
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
 * @property PayoutSelfEmployed $self_employed Данные самозанятого, который получит выплату
 * @property PayoutSelfEmployed $selfEmployed Данные самозанятого, который получит выплату
 * @property IncomeReceipt $receipt Данные чека, зарегистрированного в ФНС
 * @property CancellationDetailsInterface $cancellationDetails Комментарий к отмене выплаты
 * @property CancellationDetailsInterface $cancellation_details Комментарий к отмене выплаты
 * @property Metadata $metadata Метаданные выплаты указанные мерчантом
 * @property bool $test Признак тестовой операции
 */
class Payout extends AbstractObject implements PayoutInterface
{
    /** @var int Максимальная длина строки описания выплаты */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /** @var int Максимальная длина идентификатора выплаты */
    public const MAX_LENGTH_ID = 50;

    /** @var int Максимальная длина идентификатора выплаты */
    public const MIN_LENGTH_ID = 36;

    /**
     * Идентификатор выплаты.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_ID)]
    #[Assert\Length(min: self::MIN_LENGTH_ID)]
    protected ?string $_id = null;

    /**
     * Сумма выплаты
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    protected ?AmountInterface $_amount = null;

    /**
     * Текущее состояние выплаты
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [PayoutStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_status = null;

    /**
     * Способ проведения выплаты
     *
     * @var AbstractPayoutDestination|null
     */
    #[Assert\NotBlank]
    #[Assert\Type(AbstractPayoutDestination::class)]
    protected ?AbstractPayoutDestination $_payout_destination = null;

    /**
     * Описание транзакции (не более 128 символов). Например: «Выплата по договору 37».
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    protected ?string $_description = null;

    /**
     * Время создания выплаты. Пример: ~`2017-11-03T11:52:31.827Z`
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_created_at = null;

    /**
     * Время успешного проведения выплаты. Пример: ~`2017-11-03T11:52:42.312Z`
     * Обязательный параметр для выплат в статусе ~`succeeded`
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_succeeded_at = null;

    /**
     * Сделка, в рамках которой нужно провести выплату. Присутствует, если вы проводите Безопасную сделку
     *
     * @var PayoutDealInfo|null
     */
    #[Assert\Type(PayoutDealInfo::class)]
    protected ?PayoutDealInfo $_deal = null;

    /**
     * Данные самозанятого, который получит выплату. Присутствует, если вы делаете выплату самозанятому
     *
     * @var PayoutSelfEmployed|null
     */
    #[Assert\Type(PayoutSelfEmployed::class)]
    protected ?PayoutSelfEmployed $_self_employed = null;

    /**
     * Данные чека, зарегистрированного в ФНС. Присутствует, если вы делаете выплату самозанятому.
     *
     * @var IncomeReceipt|null
     */
    #[Assert\Type(IncomeReceipt::class)]
    protected ?IncomeReceipt $_receipt = null;

    /**
     * Комментарий к статусу canceled: кто отменил выплаты и по какой причине
     *
     * @var PayoutCancellationDetails|null
     */
    #[Assert\Type(PayoutCancellationDetails::class)]
    protected ?PayoutCancellationDetails $_cancellation_details = null;

    /**
     * Метаданные выплаты указанные мерчантом
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    protected ?Metadata $_metadata = null;

    /**
     * Признак тестовой операции
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected ?bool $_test = false;

    /**
     * Возвращает идентификатор выплаты.
     *
     * @return string|null Идентификатор выплаты
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор выплаты.
     *
     * @param string|null $id Идентификатор выплаты
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }


    /**
     * Возвращает сумму.
     *
     * @return AmountInterface|null Сумма выплаты
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает сумму выплаты.
     *
     * @param AmountInterface|array|null $amount Сумма выплаты
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает состояние выплаты.
     *
     * @return string|null Текущее состояние выплаты
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает статус выплаты
     *
     * @param string|null $status Статус выплаты
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает описание транзакции.
     *
     * @return string|null
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает описание транзакции
     *
     * @param string|null $description Описание транзакции (не более 128 символов). Например: «Выплата по договору 37».
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает используемый способ проведения выплаты.
     *
     * @return AbstractPayoutDestination|null Способ проведения выплаты
     */
    public function getPayoutDestination(): ?AbstractPayoutDestination
    {
        return $this->_payout_destination;
    }

    /**
     * Устанавливает используемый способ проведения выплаты.
     *
     * @param AbstractPayoutDestination|array|null $payout_destination Способ проведения выплаты
     *
     * @return $this
     */
    public function setPayoutDestination(mixed $payout_destination): self
    {
        if (is_array($payout_destination)) {
            $payout_destination = (new PayoutDestinationFactory)->factoryFromArray($payout_destination);
        }
        $this->_payout_destination = $this->validatePropertyValue('_payout_destination', $payout_destination);
        return $this;
    }

    /**
     * Возвращает время создания выплаты.
     *
     * @return DateTime|null Время создания выплаты
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает время создания выплаты.
     *
     * @param DateTime|string|null $created_at Время создания выплаты. Пример: ~`2017-11-03T11:52:31.827Z`
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * Возвращает время успешного проведения выплаты.
     *
     * @return DateTime|null Время успешного проведения выплаты
     */
    public function getSucceededAt(): ?DateTime
    {
        return $this->_succeeded_at;
    }

    /**
     * Устанавливает время успешного проведения выплаты.
     *
     * @param DateTime|string|null $succeeded_at Время успешного проведения выплаты. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2017-11-03T11:52:42.312Z`  Обязательный параметр для выплат в статусе ~`succeeded`.
     *
     * @return self
     */
    public function setSucceededAt(DateTime|string|null $succeeded_at = null): self
    {
        $this->_succeeded_at = $this->validatePropertyValue('_succeeded_at', $succeeded_at);
        return $this;
    }

    /**
     * Возвращает метаданные выплаты установленные мерчантом
     *
     * @return Metadata|null Метаданные выплаты указанные мерчантом
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает метаданные выплаты.
     *
     * @param Metadata|array|null $metadata Метаданные выплаты указанные мерчантом
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
     * @return PayoutCancellationDetails|null Комментарий к статусу canceled
     */
    public function getCancellationDetails(): ?PayoutCancellationDetails
    {
        return $this->_cancellation_details;
    }

    /**
     * Устанавливает комментарий к статусу canceled: кто отменил платеж и по какой причине.
     *
     * @param PayoutCancellationDetails|array|null $cancellation_details Комментарий к статусу canceled
     *
     * @return self
     */
    public function setCancellationDetails(mixed $cancellation_details = null): self
    {
        $this->_cancellation_details = $this->validatePropertyValue('_cancellation_details', $cancellation_details);
        return $this;
    }

    /**
     * Возвращает признак тестовой операции.
     *
     * @return bool|null Признак тестовой операции
     */
    public function getTest(): ?bool
    {
        return $this->_test;
    }

    /**
     * Устанавливает признак тестовой операции.
     *
     * @param bool|null $test Признак тестовой операции
     *
     * @return self
     */
    public function setTest(?bool $test = null): self
    {
        $this->_test = $this->validatePropertyValue('_test', $test);
        return $this;
    }

    /**
     * Возвращает сделку, в рамках которой нужно провести выплату.
     *
     * @return PayoutDealInfo|null Сделка, в рамках которой нужно провести выплату
     */
    public function getDeal(): ?PayoutDealInfo
    {
        return $this->_deal;
    }

    /**
     * Устанавливает сделку, в рамках которой нужно провести выплату.
     *
     * @param PayoutDealInfo|array|null $deal Сделка, в рамках которой нужно провести выплату
     *
     * @return self
     */
    public function setDeal(mixed $deal = null): self
    {
        $this->_deal = $this->validatePropertyValue('_deal', $deal);
        return $this;
    }

    /**
     * Возвращает данные самозанятого, который получит выплату.
     *
     * @return PayoutSelfEmployed|null Данные самозанятого, который получит выплату
     */
    public function getSelfEmployed(): ?PayoutSelfEmployed
    {
        return $this->_self_employed;
    }

    /**
     * Устанавливает данные самозанятого, который получит выплату.
     *
     * @param PayoutSelfEmployed|array|null $self_employed Данные самозанятого, который получит выплату
     *
     * @return self
     */
    public function setSelfEmployed(mixed $self_employed = null): self
    {
        $this->_self_employed = $this->validatePropertyValue('_self_employed', $self_employed);
        return $this;
    }

    /**
     * Возвращает данные чека, зарегистрированного в ФНС.
     *
     * @return IncomeReceipt|null Данные чека, зарегистрированного в ФНС
     */
    public function getReceipt(): ?IncomeReceipt
    {
        return $this->_receipt;
    }

    /**
     * Устанавливает данные чека, зарегистрированного в ФНС.
     *
     * @param IncomeReceipt|array|null $receipt Данные чека, зарегистрированного в ФНС
     *
     * @return self
     */
    public function setReceipt(mixed $receipt = null): self
    {
        $this->_receipt = $this->validatePropertyValue('_receipt', $receipt);
        return $this;
    }
}
