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

namespace YooKassa\Model\Receipt;

use Traversable;
use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Helpers\ProductCode;
use YooKassa\Helpers\TypeCast;
use YooKassa\Model\AmountInterface;

/**
 * Информация о товарной позиции в заказе. Используется для формирования чека.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $description Наименование товара (тег в 54 ФЗ — 1030)
 * @property float $quantity Количество (тег в 54 ФЗ — 1023)
 * @property float $amount Суммарная стоимость покупаемого товара в копейках/центах
 * @property AmountInterface $price Цена товара (тег в 54 ФЗ — 1079)
 * @property Supplier $supplier Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224)
 * @property string $agentType Тип посредника, реализующего товар или услугу
 * @property string $agent_type Тип посредника, реализующего товар или услугу
 * @property int $vatCode Ставка НДС (тег в 54 ФЗ — 1199), число 1-10
 * @property int $vat_code Ставка НДС (тег в 54 ФЗ — 1199), число 1-10
 * @property string $paymentSubject Признак предмета расчета (тег в 54 ФЗ — 1212)
 * @property string $payment_subject Признак предмета расчета (тег в 54 ФЗ — 1212)
 * @property string $paymentMode Признак способа расчета (тег в 54 ФЗ — 1214)
 * @property string $payment_mode Признак способа расчета (тег в 54 ФЗ — 1214)
 * @property string $productCode Код товара (тег в 54 ФЗ — 1162)
 * @property string $product_code Код товара (тег в 54 ФЗ — 1162)
 * @property MarkCodeInfo $markCodeInfo Код товара (тег в 54 ФЗ — 1163)
 * @property MarkCodeInfo $mark_code_info Код товара (тег в 54 ФЗ — 1163)
 * @property string $measure Мера количества предмета расчета (тег в 54 ФЗ — 2108)
 * @property string $markMode Режим обработки кода маркировки (тег в 54 ФЗ — 2102)
 * @property string $mark_mode Режим обработки кода маркировки (тег в 54 ФЗ — 2102)
 * @property MarkQuantity $markQuantity Дробное количество маркированного товара (тег в 54 ФЗ — 1291)
 * @property MarkQuantity $mark_quantity Дробное количество маркированного товара (тег в 54 ФЗ — 1291)
 * @property ListObjectInterface|IndustryDetails[] $paymentSubjectIndustryDetails Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)
 * @property ListObjectInterface|IndustryDetails[] $payment_subject_industry_details Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)
 * @property string $countryOfOriginCode Код страны происхождения товара (тег в 54 ФЗ — 1230)
 * @property string $country_of_origin_code Код страны происхождения товара (тег в 54 ФЗ — 1230)
 * @property string $customsDeclarationNumber Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231
 * @property string $customs_declaration_number Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231
 * @property string $additionalPaymentSubjectProps Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191)
 * @property string $additional_payment_subject_props Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191)
 * @property float $excise Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229)
 * @property bool $isShipping Флаг доставки
 * @property bool $is_shipping Флаг доставки
 */
class ReceiptItem extends AbstractObject implements ReceiptItemInterface
{
    /** @var int Максимальная длинна наименования товара */
    public const DESCRIPTION_MAX_LENGTH = 128;

    /** @var int Максимальная длинна дополнительного реквизита предмета расчета */
    public const ADD_PROPS_MAX_LENGTH = 64;

    /**
     * @var string|null Наименование товара (тег в 54 ФЗ — 1030)
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::DESCRIPTION_MAX_LENGTH)]
    private ?string $_description = null;

    /**
     * @var float|null Количество (тег в 54 ФЗ — 1023)
     */
    #[Assert\NotBlank]
    #[Assert\Type('float')]
    #[Assert\GreaterThan(0)]
    private ?float $_quantity = null;

    /**
     * @var string|null Мера количества предмета расчета (тег в 54 ФЗ — 2108) — единица измерения товара, например штуки, граммы.
     * Обязателен при использовании ФФД 1.2.
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [ReceiptItemMeasure::class, 'getValidValues'])]
    private ?string $_measure = null;

    /**
     * @var MarkQuantity|null Дробное количество маркированного товара (тег в 54 ФЗ — 1291)
     */
    #[Assert\Type(MarkQuantity::class)]
    #[Assert\Valid]
    private ?MarkQuantity $_mark_quantity = null;

    /**
     * @var AmountInterface|null Цена товара (тег в 54 ФЗ — 1079)
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(ReceiptItemAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * @var null|int Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199)
     */
    #[Assert\NotBlank]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    #[Assert\LessThanOrEqual(10)]
    private ?int $_vat_code = null;

    /**
     * @var null|string Признак предмета расчета (тег в 54 ФЗ — 1212)
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PaymentSubject::class, 'getValidValues'])]
    private ?string $_payment_subject = null;

    /**
     * @var string|null Признак способа расчета (тег в 54 ФЗ — 1214)
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PaymentMode::class, 'getValidValues'])]
    private ?string $_payment_mode = null;

    /**
     * @var string|null Код страны происхождения товара (тег в 54 ФЗ — 1230)
     */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: 2)]
    #[Assert\Regex(pattern: '/^[A-Z]{2}$/')]
    private ?string $_country_of_origin_code = null;

    /**
     * @var string|null Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 32)]
    #[Assert\Length(min: 1)]
    private ?string $_customs_declaration_number = null;

    /**
     * @var float|null Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229). Десятичное число с точностью до 2 символов после точки.
     */
    #[Assert\Type('float')]
    #[Assert\GreaterThan(0)]
    private ?float $_excise = null;

    /**
     * @var string|null Код товара (тег в 54 ФЗ — 1162)
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 96)]
    #[Assert\Regex(pattern: '/^[0-9A-F ]{2,96}$/')]
    private ?string $_product_code = null;

    /**
     * @var MarkCodeInfo|null Код товара (тег в 54 ФЗ — 1163).
     *                   Обязателен при использовании протокола ФФД 1.2, если товар нужно маркировать. Должно быть заполнено хотя бы одно из полей.
     */
    #[Assert\Type(MarkCodeInfo::class)]
    #[Assert\Valid]
    private ?MarkCodeInfo $_mark_code_info = null;

    /**
     * @var string|null Режим обработки кода маркировки (тег в 54 ФЗ — 2102). Должен принимать значение равное «0».
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0]{1}$/")]
    private ?string $_mark_mode = null;

    /**
     * @var IndustryDetails[]|ListObjectInterface|null Отраслевой реквизит чека (тег в 54 ФЗ — 1260)
     */
    #[Assert\Valid]
    #[Assert\AllType(IndustryDetails::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObject $_payment_subject_industry_details = null;

    /**
     * @var string|null Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191). Не более 64 символов.
     *             Можно передавать, если вы отправляете данные для формирования чека по сценарию Сначала платеж, потом чек
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::ADD_PROPS_MAX_LENGTH)]
    private ?string $_additional_payment_subject_props = null;

    /**
     * @var Supplier|null Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224)
     */
    #[Assert\Type(Supplier::class)]
    #[Assert\Valid]
    private ?Supplier $_supplier = null;

    /**
     * @var string|null Тип посредника, реализующего товар или услугу
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [AgentType::class, 'getValidValues'])]
    private ?string $_agent_type = null;

    /**
     * @var bool True если текущий айтем доставка, false если нет
     */
    #[Assert\Type('bool')]
    #[Assert\NotNull]
    private bool $_shipping = false;

    /**
     * Возвращает наименование товара.
     *
     * @return string|null Наименование товара
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает наименование товара.
     *
     * @param string|null $description Наименование товара
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает количество товара.
     *
     * @return float|null Количество купленного товара
     */
    public function getQuantity(): ?float
    {
        return $this->_quantity;
    }

    /**
     * Устанавливает количество покупаемого товара.
     *
     * @param float|null $quantity Количество
     *
     * @return self
     */
    public function setQuantity(?float $quantity): self
    {
        $this->_quantity = $this->validatePropertyValue('_quantity', $quantity);
        return $this;
    }

    /**
     * Возвращает общую стоимость покупаемого товара в копейках/центах.
     *
     * @return int Сумма стоимости покупаемого товара
     */
    public function getAmount(): int
    {
        return (int) round($this->_amount?->getIntegerValue() * $this->_quantity);
    }

    /**
     * Возвращает цену товара.
     *
     * @return AmountInterface Цена товара
     */
    public function getPrice(): AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает цену товара.
     *
     * @param AmountInterface|array $amount Цена товара
     *
     * @return self
     */
    public function setPrice(mixed $amount): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает ставку НДС
     *
     * @return null|int Ставка НДС, число 1-10, или null, если ставка не задана
     */
    public function getVatCode(): ?int
    {
        return $this->_vat_code;
    }

    /**
     * Устанавливает ставку НДС
     *
     * @param null|int $value Ставка НДС, число 1-10
     *
     * @return self
     *
     */
    public function setVatCode(?int $vat_code = null): self
    {
        $this->_vat_code = $this->validatePropertyValue('_vat_code', $vat_code);
        return $this;
    }

    /**
     * Возвращает признак предмета расчета.
     *
     * @return null|string Признак предмета расчета
     */
    public function getPaymentSubject(): ?string
    {
        return $this->_payment_subject;
    }

    /**
     * Устанавливает признак предмета расчета.
     *
     * @param null|string $payment_subject Признак предмета расчета
     *
     * @return self
     *
     */
    public function setPaymentSubject(?string $payment_subject = null): self
    {
        $this->_payment_subject = $this->validatePropertyValue('_payment_subject', $payment_subject);
        return $this;
    }

    /**
     * Возвращает признак способа расчета.
     *
     * @return null|string Признак способа расчета
     */
    public function getPaymentMode(): ?string
    {
        return $this->_payment_mode;
    }

    /**
     * Устанавливает признак способа расчета.
     *
     * @param string|null $payment_mode Признак способа расчета
     *
     * @return self
     */
    public function setPaymentMode(?string $payment_mode = null): self
    {
        $this->_payment_mode = $this->validatePropertyValue('_payment_mode', $payment_mode);
        return $this;
    }

    /**
     * Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.
     *
     * @return null|string Код товара
     */
    public function getProductCode(): ?string
    {
        return $this->_product_code;
    }

    /**
     * Устанавливает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.
     *
     * @param ProductCode|string $product_code Код товара
     *
     * @return self
     */
    public function setProductCode(mixed $product_code): self
    {
        if ($product_code instanceof ProductCode) {
            $product_code = (string) $product_code;
        }

        $this->_product_code = $this->validatePropertyValue('_product_code', $product_code);
        return $this;
    }

    /**
     * Возвращает код товара.
     *
     * @return MarkCodeInfo|null Код товара
     */
    public function getMarkCodeInfo(): ?MarkCodeInfo
    {
        return $this->_mark_code_info;
    }

    /**
     * Устанавливает код товара.
     *
     * @param array|MarkCodeInfo|null $mark_code_info Код товара
     *
     * @return self
     */
    public function setMarkCodeInfo(mixed $mark_code_info = null): self
    {
        $this->_mark_code_info = $this->validatePropertyValue('_mark_code_info', $mark_code_info);
        return $this;
    }

    /**
     * Возвращает меру количества предмета расчета.
     *
     * @return string|null Мера количества предмета расчета
     */
    public function getMeasure(): ?string
    {
        return $this->_measure;
    }

    /**
     * Устанавливает меру количества предмета расчета.
     *
     * @param string|null $measure Мера количества предмета расчета
     *
     * @return self
     */
    public function setMeasure(?string $measure): self
    {
        $this->_measure = $this->validatePropertyValue('_measure', $measure);
        return $this;
    }

    /**
     * Возвращает режим обработки кода маркировки.
     *
     * @return string|null Режим обработки кода маркировки
     */
    public function getMarkMode(): ?string
    {
        return $this->_mark_mode;
    }

    /**
     * Устанавливает режим обработки кода маркировки.
     *
     * @param string|null $mark_mode Режим обработки кода маркировки
     *
     * @return self
     */
    public function setMarkMode(?string $mark_mode = null): self
    {
        $this->_mark_mode = $this->validatePropertyValue('_mark_mode', $mark_mode);
        return $this;
    }

    /**
     * Возвращает дробное количество маркированного товара.
     *
     * @return MarkQuantity|null Дробное количество маркированного товара
     */
    public function getMarkQuantity(): ?MarkQuantity
    {
        return $this->_mark_quantity;
    }

    /**
     * Устанавливает дробное количество маркированного товара.
     *
     * @param array|MarkQuantity|null $mark_quantity Дробное количество маркированного товара
     */
    public function setMarkQuantity(mixed $mark_quantity = null): self
    {
        $this->_mark_quantity = $this->validatePropertyValue('_mark_quantity', $mark_quantity);
        return $this;
    }

    /**
     * Возвращает отраслевой реквизит чека.
     *
     * @return IndustryDetails[]|ListObjectInterface Отраслевой реквизит чека
     */
    public function getPaymentSubjectIndustryDetails(): ListObjectInterface
    {
        if ($this->_payment_subject_industry_details === null) {
            $this->_payment_subject_industry_details = new ListObject(IndustryDetails::class);
        }
        return $this->_payment_subject_industry_details;
    }

    /**
     * Устанавливает отраслевой реквизит чека.
     *
     * @param array|ListObjectInterface|null $payment_subject_industry_details Отраслевой реквизит чека
     *
     * @return self
     */
    public function setPaymentSubjectIndustryDetails(mixed $payment_subject_industry_details = null): self
    {
        $this->_payment_subject_industry_details = $this->validatePropertyValue('_payment_subject_industry_details', $payment_subject_industry_details);
        return $this;
    }

    /**
     * Возвращает дополнительный реквизит предмета расчета.
     *
     * @return string|null Дополнительный реквизит предмета расчета
     */
    public function getAdditionalPaymentSubjectProps(): ?string
    {
        return $this->_additional_payment_subject_props;
    }

    /**
     * Устанавливает дополнительный реквизит предмета расчета.
     *
     * @param string|null $additional_payment_subject_props Дополнительный реквизит предмета расчета
     */
    public function setAdditionalPaymentSubjectProps(?string $additional_payment_subject_props): void
    {
        $this->_additional_payment_subject_props = $this->validatePropertyValue(
            '_additional_payment_subject_props',
            $additional_payment_subject_props
        );
    }

    /**
     * Возвращает код страны происхождения товара по общероссийскому классификатору стран мира.
     *
     * @return null|string Код страны происхождения товара
     */
    public function getCountryOfOriginCode(): ?string
    {
        return $this->_country_of_origin_code;
    }

    /**
     * Устанавливает код страны происхождения товара по общероссийскому классификатору стран мира.
     *
     * @param string|null $country_of_origin_code Код страны происхождения товара
     *
     * @return self
     */
    public function setCountryOfOriginCode(?string $country_of_origin_code = null): self
    {
        $this->_country_of_origin_code = $this->validatePropertyValue(
            '_country_of_origin_code',
            $country_of_origin_code
        );
        return $this;
    }

    /**
     * Возвращает номер таможенной декларации.
     *
     * @return null|string Номер таможенной декларации (от 1 до 32 символов)
     */
    public function getCustomsDeclarationNumber(): ?string
    {
        return $this->_customs_declaration_number;
    }

    /**
     * Устанавливает номер таможенной декларации (от 1 до 32 символов).
     *
     * @param string|null $customs_declaration_number Номер таможенной декларации
     *
     * @return self
     *
     */
    public function setCustomsDeclarationNumber(?string $customs_declaration_number = null): self
    {
        $this->_customs_declaration_number = $this->validatePropertyValue(
            '_customs_declaration_number',
            $customs_declaration_number
        );
        return $this;
    }

    /**
     * Возвращает сумму акциза товара с учетом копеек.
     *
     * @return null|float Сумма акциза товара с учетом копеек
     */
    public function getExcise(): ?float
    {
        return $this->_excise;
    }

    /**
     * Устанавливает сумму акциза товара с учетом копеек.
     *
     * @param float|null $excise Сумма акциза товара с учетом копеек
     *
     * @return self
     *
     */
    public function setExcise(?float $excise = null): self
    {
        $this->_excise = $this->validatePropertyValue('_excise', $excise);
        return $this;
    }

    /**
     * Устанавливает флаг доставки для текущего объекта айтема в чеке.
     *
     * @param bool $value True если айтем является доставкой, false если нет
     *
     * @return self
     *
     */
    public function setIsShipping(bool $value): self
    {
        $this->_shipping = $value;
        return $this;
    }

    /**
     * Возвращает информацию о поставщике товара или услуги.
     *
     * @return Supplier|null
     */
    public function getSupplier(): ?Supplier
    {
        return $this->_supplier;
    }

    /**
     * Устанавливает информацию о поставщике товара или услуги.
     *
     * @param array|SupplierInterface|null $supplier Информация о поставщике товара или услуги
     *
     * @return self
     */
    public function setSupplier(mixed $supplier = null): self
    {
        $this->_supplier = $this->validatePropertyValue('_supplier', $supplier);
        return $this;
    }

    /**
     * Устанавливает тип посредника, реализующего товар или услугу.
     *
     * @param string|null $agent_type Тип посредника
     *
     * @return self
     */
    public function setAgentType(?string $agent_type = null): self
    {
        $this->_agent_type = $this->validatePropertyValue('_agent_type', $agent_type);
        return $this;
    }

    /**
     * Возвращает тип посредника, реализующего товар или услугу.
     *
     * @return string|null Тип посредника
     */
    public function getAgentType(): ?string
    {
        return $this->_agent_type;
    }

    /**
     * Проверяет, является ли текущий элемент чека доставкой.
     *
     * @return bool True если доставка, false если обычный товар
     */
    public function isShipping(): bool
    {
        return $this->_shipping;
    }

    /**
     * Применяет для товара скидку.
     *
     * @param float|null $coefficient Множитель скидки
     */
    public function applyDiscountCoefficient(?float $coefficient): void
    {
        $this->_amount->multiply($coefficient);
    }

    /**
     * Увеличивает цену товара на указанную величину.
     *
     * @param float|null $value Сумма на которую цену товара увеличиваем
     */
    public function increasePrice(?float $value): void
    {
        $this->_amount->increase($value);
    }

    /**
     * Уменьшает количество покупаемого товара на указанное, возвращает объект позиции в чеке с уменьшаемым количеством
     *
     * @param float|null $count Количество на которое уменьшаем позицию в чеке
     *
     * @throws EmptyPropertyValueException Выбрасывается если было передано пустое значение
     * @throws InvalidPropertyValueException Выбрасывается если в качестве аргумента был передан ноль
     *                                       или отрицательное число, или число больше текущего количества покупаемого товара
     * @throws InvalidPropertyValueTypeException Выбрасывается если в качестве аргумента было передано не число
     */
    public function fetchItem(?float $count): ReceiptItem
    {
        if (null === $count) {
            throw new EmptyPropertyValueException(
                'Empty quantity value in ReceiptItem in fetchItem method',
                0,
                'ReceiptItem.quantity'
            );
        }
        if (!is_numeric($count)) {
            throw new InvalidPropertyValueTypeException(
                'Invalid quantity value type in ReceiptItem in fetchItem method',
                0,
                'ReceiptItem.quantity',
                $count
            );
        }
        if ($count <= 0.0 || $count >= $this->_quantity) {
            throw new InvalidPropertyValueException(
                'Invalid quantity value in ReceiptItem in fetchItem method',
                0,
                'ReceiptItem.quantity',
                $count
            );
        }

        $result = clone $this;
        $result->setPrice(clone $this->getPrice());
        $result->setQuantity($count);
        $this->_quantity -= $count;

        return $result;
    }

    /**
     * Устанавливает значения свойств текущего объекта из массива.
     *
     * @param array|Traversable $sourceArray Ассоциативный массив с настройками
     */
    public function fromArray(iterable $sourceArray): void
    {
        if (isset($sourceArray['amount'])) {
            if (is_array($sourceArray['amount'])) {
                $sourceArray['price'] = new ReceiptItemAmount($sourceArray['amount']);
            } elseif ($sourceArray['amount'] instanceof AmountInterface) {
                $sourceArray['price'] = $sourceArray['amount'];
            }
            unset($sourceArray['amount']);
        }

        parent::fromArray($sourceArray);
    }

    /**
     * {@inheritdoc}
     */
    public function jsonSerialize(): array
    {
        $result = parent::jsonSerialize();

        $result['amount'] = $result['price'];
        unset($result['price'], $result['title']);

        return $result;
    }
}
