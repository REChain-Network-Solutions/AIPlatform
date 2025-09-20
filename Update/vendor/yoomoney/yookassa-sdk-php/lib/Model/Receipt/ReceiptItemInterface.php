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

use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;

/**
 * Interface ReceiptItemInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $description Наименование товара (тег в 54 ФЗ — 1030)
 * @property float $quantity Количество (тег в 54 ФЗ — 1023)
 * @property string $measure Мера количества предмета расчета (тег в 54 ФЗ — 2108)
 * @property MarkQuantity $markQuantity Дробное количество маркированного товара (тег в 54 ФЗ — 1291)
 * @property MarkQuantity $mark_quantity Дробное количество маркированного товара (тег в 54 ФЗ — 1291)
 * @property float $amount Суммарная стоимость покупаемого товара в копейках/центах
 * @property AmountInterface $price Цена товара (тег в 54 ФЗ — 1079)
 * @property int $vatCode Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199)
 * @property int $vat_code Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199)
 * @property string $paymentSubject Признак предмета расчета (тег в 54 ФЗ — 1212)
 * @property string $payment_subject Признак предмета расчета (тег в 54 ФЗ — 1212)
 * @property string $paymentMode Признак способа расчета (тег в 54 ФЗ — 1214)
 * @property string $payment_mode Признак способа расчета (тег в 54 ФЗ — 1214)
 * @property string $countryOfOriginCode Код страны происхождения товара (тег в 54 ФЗ — 1230)
 * @property string $country_of_origin_code Код страны происхождения товара (тег в 54 ФЗ — 1230)
 * @property string $customsDeclarationNumber Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231
 * @property string $customs_declaration_number Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231
 * @property float $excise Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229)
 * @property string $productCode Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162)
 * @property string $product_code Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162)
 * @property MarkCodeInfo $markCodeInfo Код товара (тег в 54 ФЗ — 1163)
 * @property MarkCodeInfo $mark_code_info Код товара (тег в 54 ФЗ — 1163)
 * @property string $markMode Режим обработки кода маркировки (тег в 54 ФЗ — 2102)
 * @property string $mark_mode Режим обработки кода маркировки (тег в 54 ФЗ — 2102)
 * @property IndustryDetails[] $paymentSubjectIndustryDetails Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)
 * @property IndustryDetails[] $payment_subject_industry_details Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)
 * @property string $additionalPaymentSubjectProps Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191)
 * @property string $additional_payment_subject_props Дополнительный реквизит предмета расчета (тег в 54 ФЗ — 1191)
 * @property Supplier $supplier Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224)
 * @property string $agentType Тип посредника, реализующего товар или услугу
 * @property string $agent_type Тип посредника, реализующего товар или услугу
 */
interface ReceiptItemInterface
{
    /**
     * Возвращает наименование товара.
     *
     * @return string|null Наименование товара
     */
    public function getDescription(): ?string;

    /**
     * Возвращает количество товара.
     *
     * @return float|null Количество купленного товара
     */
    public function getQuantity(): ?float;

    /**
     * Возвращает общую стоимость покупаемого товара в копейках/центах.
     *
     * @return int Сумма стоимости покупаемого товара
     */
    public function getAmount(): int;

    /**
     * Возвращает цену товара.
     *
     * @return AmountInterface Цена товара
     */
    public function getPrice(): AmountInterface;

    /**
     * Возвращает ставку НДС
     *
     * @return null|int Ставка НДС, число 1-10, или null, если ставка не задана
     */
    public function getVatCode(): ?int;

    /**
     * Возвращает признак предмета расчета.
     *
     * @return null|string Признак предмета расчета
     */
    public function getPaymentSubject(): ?string;

    /**
     * Возвращает признак способа расчета.
     *
     * @return null|string Признак способа расчета
     */
    public function getPaymentMode(): ?string;

    /**
     * Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.
     *
     * @return null|string Код товара
     */
    public function getProductCode(): ?string;

    /**
     * Возвращает код товара.
     *
     * @return MarkCodeInfo|null Код товара
     */
    public function getMarkCodeInfo(): ?MarkCodeInfo;

    /**
     * Возвращает меру количества предмета расчета.
     *
     * @return string|null Мера количества предмета расчета
     */
    public function getMeasure(): ?string;

    /**
     * Возвращает режим обработки кода маркировки.
     *
     * @return string|null Режим обработки кода маркировки
     */
    public function getMarkMode(): ?string;

    /**
     * Возвращает дробное количество маркированного товара.
     *
     * @return MarkQuantity|null Дробное количество маркированного товара
     */
    public function getMarkQuantity(): ?MarkQuantity;

    /**
     * Возвращает отраслевой реквизит чека.
     *
     * @return IndustryDetails[]|ListObjectInterface Отраслевой реквизит чека
     */
    public function getPaymentSubjectIndustryDetails(): ListObjectInterface;

    /**
     * Возвращает дополнительный реквизит предмета расчета.
     *
     * @return string|null Дополнительный реквизит предмета расчета
     */
    public function getAdditionalPaymentSubjectProps(): ?string;

    /**
     * Возвращает код страны происхождения товара по общероссийскому классификатору стран мира.
     *
     * @return null|string Код страны происхождения товара
     */
    public function getCountryOfOriginCode(): ?string;

    /**
     * Возвращает номер таможенной декларации.
     *
     * @return null|string Номер таможенной декларации (от 1 до 32 символов)
     */
    public function getCustomsDeclarationNumber(): ?string;

    /**
     * Возвращает сумму акциза товара с учетом копеек.
     *
     * @return null|float Сумма акциза товара с учетом копеек
     */
    public function getExcise(): ?float;

    /**
     * Возвращает информацию о поставщике товара или услуги.
     *
     * @return SupplierInterface|null Информация о поставщике товара или услуги
     */
    public function getSupplier(): ?SupplierInterface;

    /**
     * Возвращает тип посредника, реализующего товар или услугу.
     *
     * @return string|null Тип посредника, реализующего товар или услугу
     */
    public function getAgentType(): ?string;

    /**
     * Проверяет, является ли текущий элемент чека доставкой.
     *
     * @return bool True если доставка, false если обычный товар
     */
    public function isShipping(): bool;
}
