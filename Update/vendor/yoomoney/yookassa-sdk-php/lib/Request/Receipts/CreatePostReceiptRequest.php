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

use Traversable;
use YooKassa\Common\AbstractRequest;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Receipt\AdditionalUserProps;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\OperationalDetails;
use YooKassa\Model\Receipt\ReceiptCustomer;
use YooKassa\Model\Receipt\ReceiptCustomerInterface;
use YooKassa\Model\Receipt\ReceiptItem;
use YooKassa\Model\Receipt\ReceiptItemInterface;
use YooKassa\Model\Receipt\ReceiptType;
use YooKassa\Model\Receipt\Settlement;
use YooKassa\Model\Receipt\SettlementInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс объекта запроса к API на создание чека.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @example 02-builder.php 88 57 Пример использования билдера
 *
 * @property ReceiptCustomer $customer Информация о плательщике
 * @property string $type Тип чека в онлайн-кассе: приход "payment" или возврат "refund".
 * @property bool $send Формирование чека в онлайн-кассе сразу после создания объекта чека. Сейчас можно передать только значение ~`true`.
 * @property int $taxSystemCode Система налогообложения магазина (тег в 54 ФЗ — 1055).
 * @property int $tax_system_code Система налогообложения магазина (тег в 54 ФЗ — 1055).
 * @property AdditionalUserProps $additionalUserProps Дополнительный реквизит пользователя.
 * @property AdditionalUserProps $additional_user_props Дополнительный реквизит пользователя.
 * @property array $receiptIndustryDetails Отраслевой реквизит предмета расчета.
 * @property array $receipt_industry_details Отраслевой реквизит предмета расчета.
 * @property OperationalDetails $receiptOperationalDetails Операционный реквизит чека.
 * @property OperationalDetails $receipt_operational_details Операционный реквизит чека.
 * @property array $items Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.
 * @property array $settlements Список платежей
 * @property string $objectId Идентификатор объекта оплаты
 * @property string $object_id Идентификатор объекта оплаты
 * @property string $objectType Тип объекта: приход "payment" или возврат "refund".
 * @property string $object_type Тип объекта: приход "payment" или возврат "refund".
 * @property string $onBehalfOf Идентификатор магазина в ЮKassa.
 * @property string $on_behalf_of Идентификатор магазина в ЮKassa.
 */
class CreatePostReceiptRequest extends AbstractRequest implements CreatePostReceiptRequestInterface
{
    /** @var ReceiptCustomerInterface|null Информация о плательщике */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(ReceiptCustomer::class)]
    private ?ReceiptCustomerInterface $_customer = null;

    /** @var string|null Тип чека в онлайн-кассе: приход "payment" или возврат "refund". */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [ReceiptType::class, 'getValidValues'])]
    private ?string $_type = null;

    /** @var bool Признак отложенной отправки чека. */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    private bool $_send = true;

    /** @var int|null Код системы налогообложения. Число 1-6. Обязательный параметр, если вы используете онлайн-кассу Атол Онлайн, обновленную до ФФД 1.2, или у вас несколько систем налогообложения, в остальных случаях не передается. */
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    #[Assert\LessThanOrEqual(6)]
    private ?int $_tax_system_code = null;

    /** @var null|AdditionalUserProps Дополнительный реквизит пользователя */
    #[Assert\Type(AdditionalUserProps::class)]
    #[Assert\Valid]
    private ?AdditionalUserProps $_additional_user_props = null;

    /** @var null|IndustryDetails[]|ListObjectInterface Отраслевой реквизит предмета расчета. Можно передавать, если используете Чеки от ЮKassa или онлайн-кассу, обновленную до ФФД 1.2. */
    #[Assert\Valid]
    #[Assert\AllType(IndustryDetails::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObject $_receipt_industry_details = null;

    /** @var OperationalDetails|null Операционный реквизит чека */
    #[Assert\Type(OperationalDetails::class)]
    #[Assert\Valid]
    private ?OperationalDetails $_receipt_operational_details = null;

    /** @var ReceiptItemInterface[]|ListObjectInterface Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров. */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\AllType(ReceiptItem::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObject $_items = null;

    /** @var SettlementInterface[]|ListObjectInterface Список платежей */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\AllType(Settlement::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObject $_settlements = null;

    /** @var string|null Идентификатор объекта оплаты */
    private ?string $_object_id = null;

    /** @var string|null Тип объекта: приход "payment" или возврат "refund". */
    private ?string $_object_type = null;

    /** @var string|null Идентификатор магазина в ЮKassa */
    #[Assert\Type('string')]
    private ?string $_on_behalf_of = null;

    /**
     * Возвращает билдер объектов запросов создания платежа.
     *
     * @return CreatePostReceiptRequestBuilder Инстанс билдера объектов запросов
     */
    public static function builder(): CreatePostReceiptRequestBuilder
    {
        return new CreatePostReceiptRequestBuilder();
    }

    /**
     * Возвращает Id объекта чека.
     *
     * @return string|null Id объекта чека
     */
    public function getObjectId(): ?string
    {
        return $this->_object_id;
    }

    /**
     * Устанавливает Id объекта чека.
     *
     * @param string|null $value Id объекта чека
     * @return CreatePostReceiptRequest
     */
    public function setObjectId(?string $value): CreatePostReceiptRequest
    {
        $this->_object_id = $value;

        return $this;
    }

    /**
     * Возвращает тип объекта чека.
     *
     * @return string|null Тип объекта чека
     */
    public function getObjectType(): ?string
    {
        return $this->_object_type;
    }

    /**
     * Устанавливает тип объекта чека.
     *
     * @param string|null $value Тип объекта чека
     * @return CreatePostReceiptRequest
     */
    public function setObjectType(?string $value): CreatePostReceiptRequest
    {
        $this->_object_type = $value;

        return $this;
    }

    /**
     * Проверяет наличие данных о плательщике.
     */
    public function hasCustomer(): bool
    {
        return !empty($this->_customer);
    }

    /**
     * Возвращает информацию о плательщике.
     *
     * @return ReceiptCustomerInterface|null Информация о плательщике
     */
    public function getCustomer(): ?ReceiptCustomerInterface
    {
        return $this->_customer;
    }

    /**
     * Устанавливает информацию о плательщике.
     *
     * @param array|ReceiptCustomerInterface $customer Информация о плательщике
     *
     * @return self
     */
    public function setCustomer(mixed $customer = null): self
    {
        $this->_customer = $this->validatePropertyValue('_customer', $customer);
        return $this;
    }

    /**
     * Возвращает список позиций в текущем чеке.
     *
     * @return ReceiptItemInterface[]|ListObjectInterface Список товаров в заказе
     */
    public function getItems(): ListObjectInterface
    {
        if ($this->_items === null) {
            $this->_items = new ListObject(ReceiptItem::class);
        }
        return $this->_items;
    }

    /**
     * Устанавливает список позиций в чеке.
     *
     * Если до этого в чеке уже были установлены значения, они удаляются и полностью заменяются переданным списком
     * позиций. Все передаваемые значения в массиве позиций должны быть объектами класса, реализующего интерфейс
     * ReceiptItemInterface, в противном случае будет выброшено исключение InvalidPropertyValueTypeException.
     *
     * @param array|ListObjectInterface $items Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.
     *
     * @return self
     */
    public function setItems(mixed $items = null): self
    {
        $this->_items = $this->validatePropertyValue('_items', $items);
        return $this;
    }

    /**
     * Возвращает код системы налогообложения.
     *
     * @return int|null Код системы налогообложения. Число 1-6.
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
    public function setTaxSystemCode(?int $tax_system_code = null): self
    {
        $this->_tax_system_code = $this->validatePropertyValue('_tax_system_code', $tax_system_code);
        return $this;
    }

    /**
     * Возвращает дополнительный реквизит пользователя.
     *
     * @return AdditionalUserProps|null Дополнительный реквизит пользователя
     */
    public function getAdditionalUserProps(): ?AdditionalUserProps
    {
        return $this->_additional_user_props;
    }

    /**
     * Устанавливает дополнительный реквизит пользователя.
     *
     * @param null|AdditionalUserProps|array $additional_user_props Дополнительный реквизит пользователя
     *
     * @return self
     */
    public function setAdditionalUserProps(mixed $additional_user_props = null): self
    {
        $this->_additional_user_props = $this->validatePropertyValue(
            '_additional_user_props',
            $additional_user_props
        );
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
     * @param null|array|ListObjectInterface $receipt_industry_details Отраслевой реквизит чека
     *
     * @return self
     */
    public function setReceiptIndustryDetails(mixed $receipt_industry_details = null): self
    {
        $this->_receipt_industry_details = $this->validatePropertyValue(
            '_receipt_industry_details',
            $receipt_industry_details
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
            '_receipt_operational_details',
            $receipt_operational_details
        );
        return $this;
    }

    /**
     * Возвращает тип чека в онлайн-кассе.
     *
     * @return string|null Тип чека в онлайн-кассе: приход "payment" или возврат "refund"
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает тип чека в онлайн-кассе.
     *
     * @param string|null $type тип чека в онлайн-кассе: приход "payment" или возврат "refund"
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        if (!$this->_object_type) {
            $this->_object_type = $this->_type;
        }
        return $this;
    }

    /**
     * Возвращает признак отложенной отправки чека.
     *
     * @return bool Признак отложенной отправки чека
     */
    public function getSend(): bool
    {
        return $this->_send;
    }

    /**
     * Устанавливает признак отложенной отправки чека.
     *
     * @param bool $send Признак отложенной отправки чека
     *
     * @return self
     */
    public function setSend(?bool $send = null): self
    {
        $this->_send = $this->validatePropertyValue('_send', $send);
        return $this;
    }

    /**
     * Возвращает массив оплат, обеспечивающих выдачу товара.
     *
     * @return SettlementInterface[]|ListObjectInterface Массив оплат, обеспечивающих выдачу товара
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
     * @param array|ListObjectInterface|null $settlements Массив оплат, обеспечивающих выдачу товара
     *
     * @return self
     */
    public function setSettlements(mixed $settlements = null): self
    {
        $this->_settlements = $this->validatePropertyValue('_settlements', $settlements);
        return $this;
    }

    /**
     * Возвращает идентификатор магазина, от имени которого нужно отправить чек.
     * Выдается ЮKassa, отображается в разделе Продавцы личного кабинета (столбец shopId).
     * Необходимо передавать, если вы используете решение ЮKassa для платформ.
     *
     * @return string|null
     */
    public function getOnBehalfOf(): ?string
    {
        return $this->_on_behalf_of;
    }

    /**
     * Устанавливает идентификатор магазина, от имени которого нужно отправить чек.
     * Выдается ЮKassa, отображается в разделе Продавцы личного кабинета (столбец shopId).
     * Необходимо передавать, если вы используете решение ЮKassa для платформ.
     *
     * @param string|null $on_behalf_of
     *
     * @return self
     */
    public function setOnBehalfOf(?string $on_behalf_of = null): self
    {
        $this->_on_behalf_of = $this->validatePropertyValue('_on_behalf_of', $on_behalf_of);
        return $this;
    }

    /**
     * Проверяет есть ли в чеке хотя бы одна позиция товаров и оплат
     *
     * @return bool True если чек не пуст, false если в чеке нет ни одной позиции
     */
    public function notEmpty(): bool
    {
        return !$this->getItems()->isEmpty() && !$this->getSettlements()->isEmpty();
    }

    /**
     * Валидирует текущий запрос, проверяет все ли нужные свойства установлены.
     *
     * @return bool True если запрос валиден, false если нет
     */
    public function validate(): bool
    {
        if (empty($this->_customer)) {
            $this->setValidationError('Receipt customer not specified');

            return false;
        }

        if (empty($this->_type) || !ReceiptType::valueExists($this->_type)) {
            $this->setValidationError('Receipt type not specified');

            return false;
        }

        if (empty($this->_object_type)) {
            $this->setValidationError('Receipt object_type not specified');

            return false;
        }

        if (empty($this->_object_id)) {
            $this->setValidationError('Receipt object_id not specified');

            return false;
        }

        if (empty($this->_send)) { // todo: пока может быть только true
            $this->setValidationError('Receipt send not specified');

            return false;
        }

        if ($this->getSettlements()->isEmpty()) {
            $this->setValidationError('Receipt settlements not specified');

            return false;
        }

        if ($this->getItems()->isEmpty()) {
            $this->setValidationError('Receipt items not specified');

            return false;
        }

        return true;
    }
}
