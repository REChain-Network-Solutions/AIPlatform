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

namespace YooKassa\Request\Receipts;

use DateTime;
use Exception;
use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Model\Payment\ReceiptRegistrationStatus;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\OperationalDetails;
use YooKassa\Model\Receipt\ReceiptType;
use YooKassa\Model\Receipt\Settlement;
use YooKassa\Model\Receipt\SettlementInterface;

/**
 * Class AbstractReceipt.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор чека в ЮKassa.
 * @property string $type Тип чека в онлайн-кассе: приход "payment" или возврат "refund".
 * @property string $status Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled").
 * @property string $fiscalAttribute Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека.
 * @property string $objectId Идентификатор объекта чека.
 * @property string $object_id Идентификатор объекта чека.
 * @property string $fiscal_attribute Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека.
 * @property string $fiscalDocumentNumber Номер фискального документа.
 * @property string $fiscal_document_number Номер фискального документа.
 * @property string $fiscalStorageNumber Номер фискального накопителя в кассовом аппарате.
 * @property string $fiscal_storage_number Номер фискального накопителя в кассовом аппарате.
 * @property string $fiscalProviderId Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать.
 * @property string $fiscal_provider_id Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать.
 * @property DateTime $registeredAt Дата и время формирования чека в фискальном накопителе.
 * @property DateTime $registered_at Дата и время формирования чека в фискальном накопителе.
 * @property int $taxSystemCode Код системы налогообложения. Число 1-6.
 * @property int $tax_system_code Код системы налогообложения. Число 1-6.
 * @property ListObjectInterface|IndustryDetails[] $receiptIndustryDetails Отраслевой реквизит чека.
 * @property ListObjectInterface|IndustryDetails[] $receipt_industry_details Отраслевой реквизит чека.
 * @property OperationalDetails $receiptOperationalDetails Операционный реквизит чека.
 * @property OperationalDetails $receipt_operational_details Операционный реквизит чека.
 * @property ListObjectInterface|ReceiptResponseItemInterface[] $items Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.
 * @property ListObjectInterface|SettlementInterface[] $settlements Перечень совершенных расчетов.
 * @property string $onBehalfOf Идентификатор магазина.
 * @property string $on_behalf_of Идентификатор магазина.
 */
abstract class AbstractReceiptResponse extends AbstractObject implements ReceiptResponseInterface
{
    /** Длина идентификатора чека */
    public const LENGTH_RECEIPT_ID = 39;

    /** @var string|null Идентификатор чека в ЮKassa. */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::LENGTH_RECEIPT_ID)]
    protected ?string $_id = null;

    /** @var string|null Тип чека в онлайн-кассе: приход "payment" или возврат "refund". */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [ReceiptType::class, 'getValidValues'])]
    protected ?string $_type = null;

    /** @var string|null Статус доставки данных для чека в онлайн-кассу "pending", "succeeded" или "canceled". */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [ReceiptRegistrationStatus::class, 'getValidValues'])]
    protected ?string $_status = null;

    /** @var string|null Номер фискального документа. */
    #[Assert\Type('string')]
    protected ?string $_fiscal_document_number = null;

    /** @var string|null Номер фискального накопителя в кассовом аппарате. */
    #[Assert\Type('string')]
    protected ?string $_fiscal_storage_number = null;

    /** @var string|null Идентификатор объекта чека */
    #[Assert\Type('string')]
    protected ?string $_object_id = null;

    /**
     * @var string|null Фискальный признак чека.
     *             Формируется фискальным накопителем на основе данных, переданных для регистрации чека.
     */
    #[Assert\Type('string')]
    protected ?string $_fiscal_attribute = null;

    /**
     * @var DateTime|null Дата и время формирования чека в фискальном накопителе.
     *               Указывается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?\DateTime $_registered_at = null;

    /** @var string|null Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать. */
    #[Assert\Type('string')]
    protected ?string $_fiscal_provider_id = null;

    /** @var ReceiptResponseItemInterface[]|ListObjectInterface Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров */
    #[Assert\NotBlank]
    #[Assert\Type(ListObject::class)]
    #[Assert\AllType(ReceiptResponseItem::class)]
    #[Assert\Valid]
    protected ?ListObject $_items = null;

    /** @var SettlementInterface[]|ListObjectInterface Список оплат */
    #[Assert\Type(ListObject::class)]
    #[Assert\Count(min: 1)]
    #[Assert\AllType(Settlement::class)]
    #[Assert\Valid]
    protected ?ListObject $_settlements = null;

    /** @var int|null Код системы налогообложения. Число 1-6. */
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    #[Assert\LessThanOrEqual(6)]
    protected ?int $_tax_system_code = null;

    /** @var IndustryDetails[]|ListObjectInterface|null Отраслевой реквизит предмета расчета */
    #[Assert\Type(ListObject::class)]
    #[Assert\AllType(IndustryDetails::class)]
    #[Assert\Valid]
    protected ?ListObject $_receipt_industry_details = null;

    /** @var OperationalDetails|null Операционный реквизит чека */
    #[Assert\Type(OperationalDetails::class)]
    #[Assert\Valid]
    protected ?OperationalDetails $_receipt_operational_details = null;

    /** @var string|null Идентификатор магазина */
    #[Assert\Type('string')]
    protected ?string $_on_behalf_of = null;

    /**
     * AbstractReceiptResponse constructor.
     *
     * @param mixed $sourceArray
     *
     * @throws Exception
     */
    public function fromArray(iterable $sourceArray): void
    {
        parent::fromArray($sourceArray);
        if (!empty($sourceArray['refund_id']) || !empty($sourceArray['payment_id'])) {
            $this->setObjectId($this->factoryObjectId($sourceArray));
        }
        if (!empty($sourceArray['registered_at'])) {
            $this->setRegisteredAt(new DateTime($sourceArray['registered_at']));
        }
        $this->setSpecificProperties($sourceArray);
    }

    /**
     * {@inheritdoc}
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает идентификатор чека.
     *
     * @param string $id Идентификатор чека
     *
     * @return self
     */
    public function setId(string $id): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * {@inheritdoc}
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает типа чека.
     *
     * @param string $type Тип чека
     *
     * @return self
     */
    public function setType(string $type): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает идентификатор платежа или возврата, для которого был сформирован чек.
     *
     * @return string|null
     */
    public function getObjectId(): ?string
    {
        return $this->_object_id;
    }

    /**
     * Устанавливает идентификатор платежа или возврата, для которого был сформирован чек.
     *
     * @param string|null $object_id
     *
     * @return self
     */
    public function setObjectId(?string $object_id): self
    {
        $this->_object_id = $this->validatePropertyValue('_object_id', $object_id);
        return $this;
    }

    /**
     * {@inheritdoc}
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает состояние регистрации фискального чека.
     *
     * @param string|null $status Состояние регистрации фискального чека
     *
     * @return self
     */
    public function setStatus(?string $status): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает номер фискального документа.
     *
     * @return string|null Номер фискального документа
     */
    public function getFiscalDocumentNumber(): ?string
    {
        return $this->_fiscal_document_number;
    }

    /**
     * Устанавливает номер фискального документа
     *
     * @param string|null $fiscal_document_number Номер фискального документа.
     *
     * @return self
     */
    public function setFiscalDocumentNumber(?string $fiscal_document_number = null): self
    {
        $this->_fiscal_document_number = $this->validatePropertyValue(
            '_fiscal_document_number', $fiscal_document_number
        );
        return $this;
    }

    /**
     * Возвращает номер фискального накопителя в кассовом аппарате.
     *
     * @return string|null Номер фискального накопителя в кассовом аппарате
     */
    public function getFiscalStorageNumber(): ?string
    {
        return $this->_fiscal_storage_number;
    }

    /**
     * Устанавливает номер фискального накопителя в кассовом аппарате.
     *
     * @param string|null $fiscal_storage_number Номер фискального накопителя в кассовом аппарате.
     *
     * @return self
     */
    public function setFiscalStorageNumber(?string $fiscal_storage_number = null): self
    {
        $this->_fiscal_storage_number = $this->validatePropertyValue(
            '_fiscal_storage_number', $fiscal_storage_number
        );
        return $this;
    }

    /**
     * Возвращает фискальный признак чека.
     *
     * @return string|null Фискальный признак чека
     */
    public function getFiscalAttribute(): ?string
    {
        return $this->_fiscal_attribute;
    }

    /**
     * Устанавливает фискальный признак чека.
     *
     * @param string|null $fiscal_attribute Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека.
     *
     * @return self
     */
    public function setFiscalAttribute(?string $fiscal_attribute = null): self
    {
        $this->_fiscal_attribute = $this->validatePropertyValue('_fiscal_attribute', $fiscal_attribute);
        return $this;
    }

    /**
     * Возвращает дату и время формирования чека в фискальном накопителе.
     *
     * @return DateTime|null Дата и время формирования чека в фискальном накопителе
     */
    public function getRegisteredAt(): ?DateTime
    {
        return $this->_registered_at;
    }

    /**
     * Устанавливает дату и время формирования чека в фискальном накопителе.
     *
     * @param DateTime|string|null $registered_at Дата и время формирования чека в фискальном накопителе. Указывается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     *
     * @return self
     */
    public function setRegisteredAt(DateTime|string|null $registered_at = null): self
    {
        $this->_registered_at = $this->validatePropertyValue('_registered_at', $registered_at);
        return $this;
    }

    /**
     * Возвращает идентификатор чека в онлайн-кассе.
     *
     * @return string|null Идентификатор чека в онлайн-кассе
     */
    public function getFiscalProviderId(): ?string
    {
        return $this->_fiscal_provider_id;
    }

    /**
     * Устанавливает идентификатор чека в онлайн-кассе.
     *
     * @param string|null $fiscal_provider_id Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать.
     *
     * @return self
     */
    public function setFiscalProviderId(?string $fiscal_provider_id = null): self
    {
        $this->_fiscal_provider_id = $this->validatePropertyValue('_fiscal_provider_id', $fiscal_provider_id);
        return $this;
    }

    /**
     * {@inheritdoc}
     *
     * @return ReceiptResponseItemInterface[]|ListObjectInterface
     */
    public function getItems(): ListObjectInterface
    {
        if ($this->_items === null) {
            $this->_items = new ListObject(ReceiptResponseItem::class);
        }
        return $this->_items;
    }

    /**
     * Устанавливает список позиций в чеке.
     *
     * @param ReceiptResponseItemInterface[]|null $items Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров
     *
     * @return self
     */
    public function setItems(?array $items): self
    {
        $this->_items = $this->validatePropertyValue('_items', $items);
        return $this;
    }

    /**
     * Добавляет товар в чек.
     *
     * @param ReceiptResponseItemInterface $value Объект добавляемой в чек позиции
     */
    public function addItem(ReceiptResponseItemInterface $value): void
    {
        $this->_items[] = $value;
    }

    /**
     * Возвращает Массив оплат, обеспечивающих выдачу товара.
     *
     * @return SettlementInterface[]|ListObjectInterface
     */
    public function getSettlements(): ListObjectInterface
    {
        if ($this->_settlements === null) {
            $this->_settlements = new ListObject(Settlement::class);
        }
        return $this->_settlements;
    }

    /**
     * Устанавливает массив оплат, обеспечивающих выдачу товара.
     *
     * @param SettlementInterface[]|null $settlements
     *
     * @return self
     */
    public function setSettlements(?array $settlements): self
    {
        $this->_settlements = $this->validatePropertyValue('_settlements', $settlements);
        return $this;
    }

    /**
     * Добавляет оплату в массив.
     */
    public function addSettlement(SettlementInterface $value): void
    {
        $this->_settlements[] = $value;
    }

    /**
     * {@inheritdoc}
     */
    public function getTaxSystemCode(): ?int
    {
        return $this->_tax_system_code;
    }

    /**
     * Устанавливает код системы налогообложения.
     *
     * @param int|null $tax_system_code Код системы налогообложения. Число 1-6
     *
     * @return self
     */
    public function setTaxSystemCode(?int $tax_system_code): self
    {
        $this->_tax_system_code = $this->validatePropertyValue('_tax_system_code', $tax_system_code);
        return $this;
    }

    /**
     * Возвращает отраслевой реквизит чека.
     *
     * @return IndustryDetails[]|ListObjectInterface Отраслевой реквизит чека
     */
    public function getReceiptIndustryDetails(): ListObjectInterface
    {
        if ($this->_receipt_industry_details === null) {
            $this->_receipt_industry_details = new ListObject(IndustryDetails::class);
        }
        return $this->_receipt_industry_details;
    }

    /**
     * Устанавливает отраслевой реквизит чека.
     *
     * @param array|IndustryDetails[]|null $receipt_industry_details Отраслевой реквизит чека
     *
     * @return self
     */
    public function setReceiptIndustryDetails(?array $receipt_industry_details = null): self
    {
        $this->_receipt_industry_details = $this->validatePropertyValue(
            '_receipt_industry_details', $receipt_industry_details
        );
        return $this;
    }

    /**
     * Возвращает операционный реквизит чека.
     *
     * @return OperationalDetails|null Операционный реквизит чека
     */
    public function getReceiptOperationalDetails(): ?OperationalDetails
    {
        return $this->_receipt_operational_details;
    }

    /**
     * Устанавливает операционный реквизит чека.
     *
     * @param array|OperationalDetails|null $receipt_operational_details Операционный реквизит чека
     *
     * @return self
     */
    public function setReceiptOperationalDetails(mixed $receipt_operational_details = null): self
    {
        $this->_receipt_operational_details = $this->validatePropertyValue(
            '_receipt_operational_details', $receipt_operational_details
        );
        return $this;
    }

    /**
     * {@inheritdoc}
     */
    public function getOnBehalfOf(): ?string
    {
        return $this->_on_behalf_of;
    }

    /**
     * Возвращает идентификатор магазина, от имени которого нужно отправить чек.
     *
     * @param string|null $on_behalf_of Идентификатор магазина, от имени которого нужно отправить чек
     *
     * @return self
     */
    public function setOnBehalfOf(?string $on_behalf_of = null): self
    {
        $this->_on_behalf_of = $this->validatePropertyValue('_on_behalf_of', $on_behalf_of);
        return $this;
    }

    /**
     * Проверяет есть ли в чеке хотя бы одна позиция.
     *
     * @return bool True если чек не пуст, false если в чеке нет ни одной позиции
     */
    public function notEmpty(): bool
    {
        return !$this->getItems()->isEmpty();
    }

    /**
     * Установка свойств, присущих конкретному объекту.
     */
    abstract public function setSpecificProperties(array $receiptData): void;

    /**
     * Фабричный метод создания идентификатора объекта, для которого был сформирован чек.
     *
     * @param array $receiptData Массив данных чека
     */
    private function factoryObjectId(array $receiptData): ?string
    {
        if (array_key_exists('refund_id', $receiptData)) {
            return $receiptData['refund_id'];
        }
        if (array_key_exists('payment_id', $receiptData)) {
            return $receiptData['payment_id'];
        }

        return null;
    }
}
