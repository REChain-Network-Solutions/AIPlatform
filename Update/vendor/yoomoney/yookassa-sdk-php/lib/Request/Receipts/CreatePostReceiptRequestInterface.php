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

use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Receipt\AdditionalUserProps;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\OperationalDetails;
use YooKassa\Model\Receipt\ReceiptCustomer;
use YooKassa\Model\Receipt\ReceiptCustomerInterface;
use YooKassa\Model\Receipt\ReceiptItemInterface;
use YooKassa\Model\Receipt\SettlementInterface;

/**
 * Interface CreatePostReceiptRequestInterface.
 *
 * @category Interface
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $objectId Идентификатор объекта ("payment" или "refund), для которого формируется чек
 * @property string $object_id Идентификатор объекта ("payment" или "refund), для которого формируется чек
 * @property string $type Тип чека в онлайн-кассе: приход "payment" или возврат "refund"
 * @property bool $send Признак отложенной отправки чека
 * @property ReceiptCustomer $customer Информация о плательщике
 * @property int $taxSystemCode Код системы налогообложения. Число 1-6
 * @property int $tax_system_code Код системы налогообложения. Число 1-6
 * @property AdditionalUserProps $additionalUserProps Дополнительный реквизит пользователя
 * @property AdditionalUserProps $additional_user_props Дополнительный реквизит пользователя
 * @property IndustryDetails[] $receiptIndustryDetails Отраслевой реквизит чека
 * @property IndustryDetails[] $receipt_industry_details Отраслевой реквизит чека
 * @property OperationalDetails $receiptOperationalDetails Операционный реквизит чека
 * @property OperationalDetails $receipt_operational_details Операционный реквизит чека
 * @property ReceiptItemInterface[] $items Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров
 * @property SettlementInterface[] $settlements Массив оплат, обеспечивающих выдачу товара
 */
interface CreatePostReceiptRequestInterface
{
    /**
     * Возвращает идентификатор объекта, для которого формируется чек.
     *
     * @return string|null Идентификатор объекта
     */
    public function getObjectId(): ?string;

    /**
     * Устанавливает идентификатор объекта, для которого формируется чек.
     *
     * @param string|null $value Идентификатор объекта
     * @return CreatePostReceiptRequestInterface
     */
    public function setObjectId(?string $value): CreatePostReceiptRequestInterface;

    /**
     * Возвращает тип чека в онлайн-кассе.
     *
     * @return string|null Тип чека в онлайн-кассе: приход "payment" или возврат "refund"
     */
    public function getType(): ?string;

    /**
     * Устанавливает тип чека в онлайн-кассе.
     *
     * @param string $type Тип чека в онлайн-кассе: приход "payment" или возврат "refund"
     * @return CreatePostReceiptRequestInterface
     */
    public function setType(string $type): CreatePostReceiptRequestInterface;

    /**
     * Возвращает тип объекта чека.
     *
     * @return string|null Тип объекта чека
     */
    public function getObjectType(): ?string;

    /**
     * Устанавливает тип объекта чека.
     *
     * @param string|null $value Тип объекта чека
     * @return CreatePostReceiptRequestInterface
     */
    public function setObjectType(?string $value): CreatePostReceiptRequestInterface;

    /**
     * Возвращает признак отложенной отправки чека.
     *
     *  @return bool Признак отложенной отправки чека
     */
    public function getSend(): bool;

    /**
     * Устанавливает признак отложенной отправки чека.
     *
     * @param bool $send Признак отложенной отправки чека
     * @return CreatePostReceiptRequestInterface
     */
    public function setSend(bool $send): CreatePostReceiptRequestInterface;

    /**
     * Возвращает код системы налогообложения.
     *
     * @return int|null Код системы налогообложения. Число 1-6
     */
    public function getTaxSystemCode(): ?int;

    /**
     * Устанавливает код системы налогообложения.
     *
     * @param int|null $tax_system_code Код системы налогообложения. Число 1-6
     * @return CreatePostReceiptRequestInterface
     */
    public function setTaxSystemCode(?int $tax_system_code): CreatePostReceiptRequestInterface;

    /**
     * Возвращает дополнительный реквизит пользователя.
     *
     * @return AdditionalUserProps|null Дополнительный реквизит пользователя
     */
    public function getAdditionalUserProps(): ?AdditionalUserProps;

    /**
     * Устанавливает дополнительный реквизит пользователя.
     *
     * @param AdditionalUserProps|array|null $additional_user_props Дополнительный реквизит пользователя
     * @return CreatePostReceiptRequestInterface
     */
    public function setAdditionalUserProps(mixed $additional_user_props): CreatePostReceiptRequestInterface;

    /**
     * Возвращает отраслевой реквизит чека.
     *
     * @return IndustryDetails[]|ListObjectInterface Отраслевой реквизит чека
     */
    public function getReceiptIndustryDetails(): ListObjectInterface;

    /**
     * Устанавливает отраслевой реквизит чека.
     *
     * @param array|ListObjectInterface|null $receipt_industry_details Отраслевой реквизит чека
     */
    public function setReceiptIndustryDetails(mixed $receipt_industry_details);

    /**
     * Возвращает операционный реквизит чека.
     *
     * @return OperationalDetails|null Операционный реквизит чека
     */
    public function getReceiptOperationalDetails(): ?OperationalDetails;

    /**
     * Устанавливает операционный реквизит чека.
     *
     * @param array|OperationalDetails|null $receipt_operational_details Операционный реквизит чека
     */
    public function setReceiptOperationalDetails(mixed $receipt_operational_details);

    /**
     * Возвращает информацию о плательщике.
     *
     * @return ReceiptCustomerInterface|null Информация о плательщике
     */
    public function getCustomer(): ?ReceiptCustomerInterface;

    /**
     * Устанавливает информацию о пользователе.
     *
     * @param ReceiptCustomerInterface|array|null $customer Информация о плательщике
     * @return CreatePostReceiptRequestInterface
     */
    public function setCustomer(mixed $customer): CreatePostReceiptRequestInterface;

    /**
     * Возвращает Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.
     *
     *  @return ReceiptItemInterface[]|ListObjectInterface|null
     */
    public function getItems(): ?ListObjectInterface;

    /**
     * Устанавливает список товаров чека.
     *
     * @param array|ListObjectInterface|null $items Список товаров чека
     */
    public function setItems(mixed $items): CreatePostReceiptRequestInterface;

    /**
     * Возвращает Массив оплат, обеспечивающих выдачу товара.
     *
     *  @return SettlementInterface[]|ListObjectInterface|null
     */
    public function getSettlements(): ?ListObjectInterface;

    /**
     * Устанавливает массив оплат, обеспечивающих выдачу товара.
     *
     * @param array|ListObjectInterface|null $settlements Массив оплат, обеспечивающих выдачу товара
     */
    public function setSettlements(mixed $settlements): CreatePostReceiptRequestInterface;

    /**
     * Возвращает идентификатор магазина, от имени которого нужно отправить чек.
     *
     * @return null|string Идентификатор магазина, от имени которого нужно отправить чек
     */
    public function getOnBehalfOf(): ?string;

    /**
     * Устанавливает идентификатор магазина, от имени которого нужно отправить чек.
     * Выдается ЮKassa, отображается в разделе Продавцы личного кабинета (столбец shopId).
     * Необходимо передавать, если вы используете решение ЮKassa для платформ.
     *
     * @param string|null $on_behalf_of Идентификатор магазина, от имени которого нужно отправить чек
     */
    public function setOnBehalfOf(?string $on_behalf_of);

    /**
     * Проверяет есть ли в чеке хотя бы одна позиция.
     *
     * @return bool True если чек не пуст, false если в чеке нет ни одной позиции
     */
    public function notEmpty(): bool;
}
