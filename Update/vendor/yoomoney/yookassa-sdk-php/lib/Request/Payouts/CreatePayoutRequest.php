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

use YooKassa\Common\AbstractRequest;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\PayoutDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData;
use YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataFactory;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель CreatePayoutRequest.
 *
 * Класс объекта запроса к API на проведение новой выплаты.
 *
 * @example 02-builder.php 182 26 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property AmountInterface $amount Сумма создаваемой выплаты
 * @property AbstractPayoutDestinationData $payoutDestinationData Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token.
 * @property AbstractPayoutDestinationData $payout_destination_data Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token.
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
 * @property ListObjectInterface|PayoutPersonalData[] $personal_data Персональные данные получателя выплаты. Только для обычных выплат. Необходимо передавать в этих сценариях: - [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП); - [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send). В массиве можно одновременно передать несколько идентификаторов, но только для разных типов данных.
 * @property ListObjectInterface|PayoutPersonalData[] $personalData Персональные данные получателя выплаты. Только для обычных выплат. Необходимо передавать в этих сценариях: - [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП); - [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send). В массиве можно одновременно передать несколько идентификаторов, но только для разных типов данных.
 * @property Metadata $metadata Метаданные привязанные к выплате
 */
class CreatePayoutRequest extends AbstractRequest implements CreatePayoutRequestInterface
{
    /** @var int Максимальная длина строки описания сделки. */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /** @var int Максимальное количество объектов для персональных данных. */
    public const MAX_PERSONAL_DATA = 2;

    /** @var int Минимальное количество объектов для персональных данных. */
    public const MIN_PERSONAL_DATA = 1;

    /**
     * Сумма создаваемой выплаты
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Данные платежного средства, на которое нужно сделать выплату
     *
     * @var AbstractPayoutDestinationData|null
     */
    #[Assert\Valid]
    #[Assert\Type(AbstractPayoutDestinationData::class)]
    private ?AbstractPayoutDestinationData $_payout_destination_data = null;

    /**
     * Токенизированные данные для выплаты. Например, синоним банковской карты.
     * Обязательный параметр, если не передан `payout_destination_data` или `payment_method_id`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_payout_token = null;

    /**
     * Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты.
     * [Подробнее о выплатах с использованием идентификатора сохраненного способа оплаты](https://yookassa.ru/developers/payouts/scenario-extensions/multipurpose-token)
     * Обязательный параметр, если не передан payout_destination_data или payout_token.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_payment_method_id = null;

    /**
     * Сделка, в рамках которой нужно провести выплату
     *
     * @var PayoutDealInfo|null
     */
    #[Assert\Valid]
    #[Assert\Type(PayoutDealInfo::class)]
    private ?PayoutDealInfo $_deal = null;

    /**
     * Описание транзакции (не более 128 символов). Например: «Выплата по договору 37».
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    private ?string $_description = null;

    /**
     * Данные самозанятого, который получит выплату. Необходимо передавать,
     * если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.
     *
     * @var PayoutSelfEmployedInfo|null
     */
    #[Assert\Valid]
    #[Assert\Type(PayoutSelfEmployedInfo::class)]
    private ?PayoutSelfEmployedInfo $_self_employed = null;

    /**
     * Данные для формирования чека в сервисе Мой налог. Необходимо передавать,
     * если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат.
     *
     * @var IncomeReceiptData|null
     */
    #[Assert\Valid]
    #[Assert\Type(IncomeReceiptData::class)]
    private ?IncomeReceiptData $_receipt_data = null;

    /**
     * Персональные данные получателя выплаты. Только для обычных выплат.
     *
     * Необходимо передавать в этих сценариях:
     * - [выплаты с проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП);
     * - [выплаты с передачей данных получателя для выписок из реестра](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-data-send).
     *
     * В массиве можно одновременно передать несколько идентификаторов, но только для разных типов данных.
     *
     * @var PayoutPersonalData[]|ListObjectInterface|null
     */
    #[Assert\Valid]
    #[Assert\AllType(PayoutPersonalData::class)]
    #[Assert\Type(ListObject::class)]
    #[Assert\Count(max: self::MAX_PERSONAL_DATA)]
    #[Assert\Count(min: self::MIN_PERSONAL_DATA)]
    private ?ListObject $_personal_data = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    private ?Metadata $_metadata = null;

    /**
     * Возвращает сумму выплаты.
     *
     * @return AmountInterface|null Сумма выплаты
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Проверяет, была ли установлена сумма выплаты.
     *
     * @return bool True если сумма выплаты была установлена, false если нет
     */
    public function hasAmount(): bool
    {
        return !empty($this->_amount);
    }

    /**
     * Устанавливает сумму выплаты.
     *
     * @param AmountInterface|array|null $amount Сумма выплаты
     *
     * @return self
     */
    public function setAmount(mixed $amount): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает данные для создания метода оплаты.
     *
     * @return AbstractPayoutDestinationData|null Данные используемые для создания метода оплаты
     */
    public function getPayoutDestinationData(): ?AbstractPayoutDestinationData
    {
        return $this->_payout_destination_data;
    }

    /**
     * Проверяет установлен ли объект с методом оплаты.
     *
     * @return bool True если объект метода оплаты установлен, false если нет
     */
    public function hasPayoutDestinationData(): bool
    {
        return !empty($this->_payout_destination_data);
    }

    /**
     * Устанавливает объект с информацией для создания метода оплаты.
     *
     * @param AbstractPayoutDestinationData|array|null $payout_destination_data Объект создания метода оплаты или null
     *
     * @return self
     */
    public function setPayoutDestinationData($payout_destination_data): self
    {
        if (is_array($payout_destination_data)) {
            $payout_destination_data = (new PayoutDestinationDataFactory)->factoryFromArray($payout_destination_data);
        }
        $this->_payout_destination_data = $this->validatePropertyValue('_payout_destination_data', $payout_destination_data);
        return $this;
    }

    /**
     * Возвращает токенизированные данные для выплаты.
     *
     * @return string|null Токенизированные данные для выплаты
     */
    public function getPayoutToken(): ?string
    {
        return $this->_payout_token;
    }

    /**
     * Проверяет наличие токенизированных данных для выплаты.
     *
     * @return bool True если токен установлен, false если нет
     */
    public function hasPayoutToken(): bool
    {
        return !empty($this->_payout_token);
    }

    /**
     * Устанавливает токенизированные данные для выплаты.
     *
     * @param string|null $payout_token Токенизированные данные для выплаты
     *
     * @return self
     */
    public function setPayoutToken(?string $payout_token): self
    {
        $this->_payout_token = $this->validatePropertyValue('_payout_token', $payout_token);
        return $this;
    }

    /**
     * Возвращает идентификатор сохраненного способа оплаты.
     *
     * @return null|string Идентификатор сохраненного способа оплаты
     */
    public function getPaymentMethodId(): ?string
    {
        return $this->_payment_method_id;
    }

    /**
     * Проверяет наличие идентификатора сохраненного способа оплаты.
     *
     * @return bool True если идентификатора установлен, false если нет
     */
    public function hasPaymentMethodId(): bool
    {
        return !empty($this->_payment_method_id);
    }

    /**
     * Устанавливает идентификатор сохраненного способа оплаты.
     *
     * @param null|string $payment_method_id Идентификатор сохраненного способа оплаты
     *
     * @return self
     */
    public function setPaymentMethodId(?string $payment_method_id = null): self
    {
        $this->_payment_method_id = $this->validatePropertyValue('_payment_method_id', $payment_method_id);
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
     * Проверяет наличие сделки в создаваемой выплате.
     *
     * @return bool True если сделка есть, false если нет
     */
    public function hasDeal(): bool
    {
        return !empty($this->_deal);
    }

    /**
     * Устанавливает сделку, в рамках которой нужно провести выплату.
     *
     * @param PayoutDealInfo|array|null $deal Сделка, в рамках которой нужно провести выплату
     *
     * @return self
     */
    public function setDeal(mixed $deal): self
    {
        $this->_deal = $this->validatePropertyValue('_deal', $deal);
        return $this;
    }

    /**
     * Возвращает данные самозанятого, который получит выплату.
     *
     * @return null|PayoutSelfEmployedInfo Данные самозанятого, который получит выплату
     */
    public function getSelfEmployed(): ?PayoutSelfEmployedInfo
    {
        return $this->_self_employed;
    }

    /**
     * Проверяет наличие данных самозанятого в создаваемой выплате.
     *
     * @return bool True если данные самозанятого есть, false если нет
     */
    public function hasSelfEmployed(): bool
    {
        return !empty($this->_self_employed);
    }

    /**
     * Устанавливает данные самозанятого, который получит выплату.
     *
     * @param PayoutSelfEmployedInfo|array|null $self_employed Данные самозанятого, который получит выплату
     *
     * @return self
     */
    public function setSelfEmployed(mixed $self_employed = null): self
    {
        $this->_self_employed = $this->validatePropertyValue('_self_employed', $self_employed);
        return $this;
    }

    /**
     * Возвращает данные для формирования чека в сервисе Мой налог.
     *
     * @return null|IncomeReceiptData Данные для формирования чека в сервисе Мой налог
     */
    public function getReceiptData(): ?IncomeReceiptData
    {
        return $this->_receipt_data;
    }

    /**
     * Проверяет наличие данных для формирования чека в сервисе Мой налог.
     *
     * @return bool True если данные для формирования чека есть, false если нет
     */
    public function hasReceiptData(): bool
    {
        return !empty($this->_receipt_data);
    }

    /**
     * Устанавливает данные для формирования чека в сервисе Мой налог.
     *
     * @param IncomeReceiptData|array|null $receipt_data Данные для формирования чека в сервисе Мой налог
     *
     * @return $this
     */
    public function setReceiptData(mixed $receipt_data = null): self
    {
        $this->_receipt_data = $this->validatePropertyValue('_receipt_data', $receipt_data);
        return $this;
    }

    /**
     * Возвращает персональные данные получателя выплаты.
     *
     * @return PayoutPersonalData[]|ListObjectInterface Персональные данные получателя выплаты
     */
    public function getPersonalData(): ListObjectInterface
    {
        if ($this->_personal_data === null) {
            $this->_personal_data = new ListObject(PayoutPersonalData::class);
        }
        return $this->_personal_data;
    }

    /**
     * Проверяет наличие персональных данных в создаваемой выплате.
     *
     * @return bool True если персональные данные есть, false если нет
     */
    public function hasPersonalData(): bool
    {
        return !empty($this->_personal_data);
    }

    /**
     * Устанавливает персональные данные получателя выплаты.
     *
     * @param ListObjectInterface|array|null $personal_data Персональные данные получателя выплаты
     *
     * @return self
     */
    public function setPersonalData(mixed $personal_data = null): self
    {
        $this->_personal_data = $this->validatePropertyValue('_personal_data', $personal_data);
        return $this;
    }

    /**
     * Возвращает описание транзакции.
     *
     * @return string|null Описание транзакции
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Проверяет наличие описания транзакции в создаваемом платеже.
     *
     * @return bool True если описание транзакции есть, false если нет
     */
    public function hasDescription(): bool
    {
        return null !== $this->_description;
    }

    /**
     * Устанавливает описание транзакции.
     *
     * @param string|null $description Описание транзакции
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает данные оплаты установленные мерчантом.
     *
     * @return Metadata|null Метаданные, привязанные к выплате
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Проверяет, были ли установлены метаданные выплаты.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool
    {
        return !empty($this->_metadata) && $this->_metadata->count() > 0;
    }

    /**
     * Устанавливает метаданные, привязанные к выплате.
     *
     * @param Metadata|array|null $metadata Метаданные выплаты, устанавливаемые мерчантом
     *
     * @return $this
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

    /**
     * Проверяет на валидность текущий объект
     *
     * @return bool True если объект запроса валиден, false если нет
     */
    public function validate(): bool
    {
        if (!$this->hasAmount()) {
            $this->setValidationError('Amount field is required');

            return false;
        }
        if (
            ($this->hasPayoutToken() && $this->hasPayoutDestinationData()) ||
            ($this->hasPayoutToken() && $this->hasPaymentMethodId()) ||
            ($this->hasPayoutDestinationData() && $this->hasPaymentMethodId())
        ) {
            $this->setValidationError('Both payoutToken, payoutDestinationData and paymentMethodId values are specified');

            return false;
        }

        if (!$this->hasPayoutToken() && !$this->hasPayoutDestinationData() && !$this->hasPaymentMethodId()) {
            $this->setValidationError('Both payoutToken, payoutDestinationData and paymentMethodId values are not specified');

            return false;
        }

        return true;
    }

    /**
     * Возвращает билдер объектов запросов создания выплаты.
     *
     * @return CreatePayoutRequestBuilder Инстанс билдера объектов запросов
     */
    public static function builder(): CreatePayoutRequestBuilder
    {
        return new CreatePayoutRequestBuilder();
    }
}
