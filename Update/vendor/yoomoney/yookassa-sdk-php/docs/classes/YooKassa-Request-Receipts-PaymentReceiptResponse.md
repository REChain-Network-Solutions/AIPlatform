# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Receipts\PaymentReceiptResponse
### Namespace: [\YooKassa\Request\Receipts](../namespaces/yookassa-request-receipts.md)
---
**Summary:**

Класс описывающий чек, привязанный к платежу.


---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [LENGTH_RECEIPT_ID](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#constant_LENGTH_RECEIPT_ID) |  | Длина идентификатора чека |
| public | [LENGTH_PAYMENT_ID](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md#constant_LENGTH_PAYMENT_ID) |  | Длина идентификатора платежа |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$fiscal_attribute](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscal_attribute) |  | Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека. |
| public | [$fiscal_document_number](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscal_document_number) |  | Номер фискального документа. |
| public | [$fiscal_provider_id](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscal_provider_id) |  | Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать. |
| public | [$fiscal_storage_number](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscal_storage_number) |  | Номер фискального накопителя в кассовом аппарате. |
| public | [$fiscalAttribute](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscalAttribute) |  | Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека. |
| public | [$fiscalDocumentNumber](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscalDocumentNumber) |  | Номер фискального документа. |
| public | [$fiscalProviderId](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscalProviderId) |  | Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать. |
| public | [$fiscalStorageNumber](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_fiscalStorageNumber) |  | Номер фискального накопителя в кассовом аппарате. |
| public | [$id](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_id) |  | Идентификатор чека в ЮKassa. |
| public | [$items](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_items) |  | Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров. |
| public | [$object_id](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_object_id) |  | Идентификатор объекта чека. |
| public | [$objectId](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_objectId) |  | Идентификатор объекта чека. |
| public | [$on_behalf_of](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_on_behalf_of) |  | Идентификатор магазина. |
| public | [$onBehalfOf](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_onBehalfOf) |  | Идентификатор магазина. |
| public | [$payment_id](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md#property_payment_id) |  | Идентификатор платежа в ЮKassa |
| public | [$paymentId](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md#property_paymentId) |  | Идентификатор платежа в ЮKassa |
| public | [$receipt_industry_details](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_receipt_industry_details) |  | Отраслевой реквизит чека. |
| public | [$receipt_operational_details](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_receipt_operational_details) |  | Операционный реквизит чека. |
| public | [$receiptIndustryDetails](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_receiptIndustryDetails) |  | Отраслевой реквизит чека. |
| public | [$receiptOperationalDetails](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_receiptOperationalDetails) |  | Операционный реквизит чека. |
| public | [$registered_at](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_registered_at) |  | Дата и время формирования чека в фискальном накопителе. |
| public | [$registeredAt](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_registeredAt) |  | Дата и время формирования чека в фискальном накопителе. |
| public | [$settlements](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_settlements) |  | Перечень совершенных расчетов. |
| public | [$status](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_status) |  | Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled"). |
| public | [$tax_system_code](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_tax_system_code) |  | Код системы налогообложения. Число 1-6. |
| public | [$taxSystemCode](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_taxSystemCode) |  | Код системы налогообложения. Число 1-6. |
| public | [$type](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property_type) |  | Тип чека в онлайн-кассе: приход "payment" или возврат "refund". |
| protected | [$_fiscal_attribute](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__fiscal_attribute) |  |  |
| protected | [$_fiscal_document_number](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__fiscal_document_number) |  |  |
| protected | [$_fiscal_provider_id](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__fiscal_provider_id) |  |  |
| protected | [$_fiscal_storage_number](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__fiscal_storage_number) |  |  |
| protected | [$_id](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__id) |  |  |
| protected | [$_items](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__items) |  |  |
| protected | [$_object_id](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__object_id) |  |  |
| protected | [$_on_behalf_of](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__on_behalf_of) |  |  |
| protected | [$_receipt_industry_details](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__receipt_industry_details) |  |  |
| protected | [$_receipt_operational_details](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__receipt_operational_details) |  |  |
| protected | [$_registered_at](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__registered_at) |  |  |
| protected | [$_settlements](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__settlements) |  |  |
| protected | [$_status](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__status) |  |  |
| protected | [$_tax_system_code](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__tax_system_code) |  |  |
| protected | [$_type](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#property__type) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [addItem()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_addItem) |  | Добавляет товар в чек. |
| public | [addSettlement()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_addSettlement) |  | Добавляет оплату в массив. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getFiscalAttribute()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getFiscalAttribute) |  | Возвращает фискальный признак чека. |
| public | [getFiscalDocumentNumber()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getFiscalDocumentNumber) |  | Возвращает номер фискального документа. |
| public | [getFiscalProviderId()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getFiscalProviderId) |  | Возвращает идентификатор чека в онлайн-кассе. |
| public | [getFiscalStorageNumber()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getFiscalStorageNumber) |  | Возвращает номер фискального накопителя в кассовом аппарате. |
| public | [getId()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getId) |  | Возвращает идентификатор чека в ЮKassa. |
| public | [getItems()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getItems) |  | Возвращает Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров. |
| public | [getObjectId()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getObjectId) |  | Возвращает идентификатор платежа или возврата, для которого был сформирован чек. |
| public | [getOnBehalfOf()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getOnBehalfOf) |  | Возвращает идентификатор магазин |
| public | [getPaymentId()](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md#method_getPaymentId) |  | Возвращает идентификатор платежа. |
| public | [getReceiptIndustryDetails()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getReceiptIndustryDetails) |  | Возвращает отраслевой реквизит чека. |
| public | [getReceiptOperationalDetails()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getReceiptOperationalDetails) |  | Возвращает операционный реквизит чека. |
| public | [getRegisteredAt()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getRegisteredAt) |  | Возвращает дату и время формирования чека в фискальном накопителе. |
| public | [getSettlements()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getSettlements) |  | Возвращает Массив оплат, обеспечивающих выдачу товара. |
| public | [getStatus()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getStatus) |  | Возвращает статус доставки данных для чека в онлайн-кассу. |
| public | [getTaxSystemCode()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getTaxSystemCode) |  | Возвращает код системы налогообложения. |
| public | [getType()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_getType) |  | Возвращает тип чека в онлайн-кассе. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [notEmpty()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_notEmpty) |  | Проверяет есть ли в чеке хотя бы одна позиция. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setFiscalAttribute()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setFiscalAttribute) |  | Устанавливает фискальный признак чека. |
| public | [setFiscalDocumentNumber()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setFiscalDocumentNumber) |  | Устанавливает номер фискального документа |
| public | [setFiscalProviderId()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setFiscalProviderId) |  | Устанавливает идентификатор чека в онлайн-кассе. |
| public | [setFiscalStorageNumber()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setFiscalStorageNumber) |  | Устанавливает номер фискального накопителя в кассовом аппарате. |
| public | [setId()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setId) |  | Устанавливает идентификатор чека. |
| public | [setItems()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setItems) |  | Устанавливает список позиций в чеке. |
| public | [setObjectId()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setObjectId) |  | Устанавливает идентификатор платежа или возврата, для которого был сформирован чек. |
| public | [setOnBehalfOf()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setOnBehalfOf) |  | Возвращает идентификатор магазина, от имени которого нужно отправить чек. |
| public | [setPaymentId()](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md#method_setPaymentId) |  | Устанавливает идентификатор платежа в ЮKassa. |
| public | [setReceiptIndustryDetails()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setReceiptIndustryDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setReceiptOperationalDetails()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setReceiptOperationalDetails) |  | Устанавливает операционный реквизит чека. |
| public | [setRegisteredAt()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setRegisteredAt) |  | Устанавливает дату и время формирования чека в фискальном накопителе. |
| public | [setSettlements()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setSettlements) |  | Устанавливает массив оплат, обеспечивающих выдачу товара. |
| public | [setSpecificProperties()](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md#method_setSpecificProperties) |  | Установка свойств, присущих конкретному объекту. |
| public | [setStatus()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setStatus) |  | Устанавливает состояние регистрации фискального чека. |
| public | [setTaxSystemCode()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setTaxSystemCode) |  | Устанавливает код системы налогообложения. |
| public | [setType()](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md#method_setType) |  | Устанавливает типа чека. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Receipts/PaymentReceiptResponse.php](../../lib/Request/Receipts/PaymentReceiptResponse.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)
  * \YooKassa\Request\Receipts\PaymentReceiptResponse

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_LENGTH_RECEIPT_ID" class="anchor"></a>
###### LENGTH_RECEIPT_ID
Inherited from [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

Длина идентификатора чека

```php
LENGTH_RECEIPT_ID = 39
```


<a name="constant_LENGTH_PAYMENT_ID" class="anchor"></a>
###### LENGTH_PAYMENT_ID
Длина идентификатора платежа

```php
LENGTH_PAYMENT_ID = 36
```



---
## Properties
<a name="property_fiscal_attribute"></a>
#### public $fiscal_attribute : string
---
***Description***

Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_fiscal_document_number"></a>
#### public $fiscal_document_number : string
---
***Description***

Номер фискального документа.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_fiscal_provider_id"></a>
#### public $fiscal_provider_id : string
---
***Description***

Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_fiscal_storage_number"></a>
#### public $fiscal_storage_number : string
---
***Description***

Номер фискального накопителя в кассовом аппарате.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_fiscalAttribute"></a>
#### public $fiscalAttribute : string
---
***Description***

Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_fiscalDocumentNumber"></a>
#### public $fiscalDocumentNumber : string
---
***Description***

Номер фискального документа.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_fiscalProviderId"></a>
#### public $fiscalProviderId : string
---
***Description***

Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_fiscalStorageNumber"></a>
#### public $fiscalStorageNumber : string
---
***Description***

Номер фискального накопителя в кассовом аппарате.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_id"></a>
#### public $id : string
---
***Description***

Идентификатор чека в ЮKassa.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_items"></a>
#### public $items : \YooKassa\Common\ListObjectInterface|\YooKassa\Request\Receipts\ReceiptResponseItemInterface[]
---
***Description***

Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Receipts\ReceiptResponseItemInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Request\Receipts\ReceiptResponseItemInterface[]">ReceiptResponseItemInterface[]</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_object_id"></a>
#### public $object_id : string
---
***Description***

Идентификатор объекта чека.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_objectId"></a>
#### public $objectId : string
---
***Description***

Идентификатор объекта чека.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_on_behalf_of"></a>
#### public $on_behalf_of : string
---
***Description***

Идентификатор магазина.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_onBehalfOf"></a>
#### public $onBehalfOf : string
---
***Description***

Идентификатор магазина.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_payment_id"></a>
#### public $payment_id : string
---
***Description***

Идентификатор платежа в ЮKassa

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_paymentId"></a>
#### public $paymentId : string
---
***Description***

Идентификатор платежа в ЮKassa

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_receipt_industry_details"></a>
#### public $receipt_industry_details : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит чека.

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_receipt_operational_details"></a>
#### public $receipt_operational_details : \YooKassa\Model\Receipt\OperationalDetails
---
***Description***

Операционный реквизит чека.

**Type:** <a href="../classes/YooKassa-Model-Receipt-OperationalDetails.html"><abbr title="\YooKassa\Model\Receipt\OperationalDetails">OperationalDetails</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_receiptIndustryDetails"></a>
#### public $receiptIndustryDetails : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит чека.

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_receiptOperationalDetails"></a>
#### public $receiptOperationalDetails : \YooKassa\Model\Receipt\OperationalDetails
---
***Description***

Операционный реквизит чека.

**Type:** <a href="../classes/YooKassa-Model-Receipt-OperationalDetails.html"><abbr title="\YooKassa\Model\Receipt\OperationalDetails">OperationalDetails</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_registered_at"></a>
#### public $registered_at : \DateTime
---
***Description***

Дата и время формирования чека в фискальном накопителе.

**Type:** \DateTime

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_registeredAt"></a>
#### public $registeredAt : \DateTime
---
***Description***

Дата и время формирования чека в фискальном накопителе.

**Type:** \DateTime

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_settlements"></a>
#### public $settlements : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\SettlementInterface[]
---
***Description***

Перечень совершенных расчетов.

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\SettlementInterface[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\SettlementInterface[]">SettlementInterface[]</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_status"></a>
#### public $status : string
---
***Description***

Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled").

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_tax_system_code"></a>
#### public $tax_system_code : int
---
***Description***

Код системы налогообложения. Число 1-6.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_taxSystemCode"></a>
#### public $taxSystemCode : int
---
***Description***

Код системы налогообложения. Число 1-6.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип чека в онлайн-кассе: приход "payment" или возврат "refund".

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__fiscal_attribute"></a>
#### protected $_fiscal_attribute : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Фискальный признак чека.
Формируется фискальным накопителем на основе данных, переданных для регистрации чека.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__fiscal_document_number"></a>
#### protected $_fiscal_document_number : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Номер фискального документа.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__fiscal_provider_id"></a>
#### protected $_fiscal_provider_id : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__fiscal_storage_number"></a>
#### protected $_fiscal_storage_number : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Номер фискального накопителя в кассовом аппарате.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__id"></a>
#### protected $_id : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Идентификатор чека в ЮKassa.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__items"></a>
#### protected $_items : ?\YooKassa\Common\ListObject
---
**Type:** <a href="../?\YooKassa\Common\ListObject"><abbr title="?\YooKassa\Common\ListObject">ListObject</abbr></a>
Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__object_id"></a>
#### protected $_object_id : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Идентификатор объекта чека
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__on_behalf_of"></a>
#### protected $_on_behalf_of : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Идентификатор магазина
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__receipt_industry_details"></a>
#### protected $_receipt_industry_details : ?\YooKassa\Common\ListObject
---
**Type:** <a href="../?\YooKassa\Common\ListObject"><abbr title="?\YooKassa\Common\ListObject">ListObject</abbr></a>
Отраслевой реквизит предмета расчета
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__receipt_operational_details"></a>
#### protected $_receipt_operational_details : ?\YooKassa\Model\Receipt\OperationalDetails
---
**Type:** <a href="../?\YooKassa\Model\Receipt\OperationalDetails"><abbr title="?\YooKassa\Model\Receipt\OperationalDetails">OperationalDetails</abbr></a>
Операционный реквизит чека
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__registered_at"></a>
#### protected $_registered_at : ?\DateTime
---
**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>
Дата и время формирования чека в фискальном накопителе.
Указывается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__settlements"></a>
#### protected $_settlements : ?\YooKassa\Common\ListObject
---
**Type:** <a href="../?\YooKassa\Common\ListObject"><abbr title="?\YooKassa\Common\ListObject">ListObject</abbr></a>
Список оплат
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__status"></a>
#### protected $_status : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Статус доставки данных для чека в онлайн-кассу &quot;pending&quot;, &quot;succeeded&quot; или &quot;canceled&quot;.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__tax_system_code"></a>
#### protected $_tax_system_code : ?int
---
**Type:** <a href="../?int"><abbr title="?int">?int</abbr></a>
Код системы налогообложения. Число 1-6.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Тип чека в онлайн-кассе: приход &quot;payment&quot; или возврат &quot;refund&quot;.
**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_addItem" class="anchor"></a>
#### public addItem() : void

```php
public addItem(\YooKassa\Request\Receipts\ReceiptResponseItemInterface $value) : void
```

**Summary**

Добавляет товар в чек.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Receipts\ReceiptResponseItemInterface</code> | value  | Объект добавляемой в чек позиции |

**Returns:** void - 


<a name="method_addSettlement" class="anchor"></a>
#### public addSettlement() : void

```php
public addSettlement(\YooKassa\Model\Receipt\SettlementInterface $value) : void
```

**Summary**

Добавляет оплату в массив.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\SettlementInterface</code> | value  |  |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getFiscalAttribute" class="anchor"></a>
#### public getFiscalAttribute() : string|null

```php
public getFiscalAttribute() : string|null
```

**Summary**

Возвращает фискальный признак чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - Фискальный признак чека


<a name="method_getFiscalDocumentNumber" class="anchor"></a>
#### public getFiscalDocumentNumber() : string|null

```php
public getFiscalDocumentNumber() : string|null
```

**Summary**

Возвращает номер фискального документа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - Номер фискального документа


<a name="method_getFiscalProviderId" class="anchor"></a>
#### public getFiscalProviderId() : string|null

```php
public getFiscalProviderId() : string|null
```

**Summary**

Возвращает идентификатор чека в онлайн-кассе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - Идентификатор чека в онлайн-кассе


<a name="method_getFiscalStorageNumber" class="anchor"></a>
#### public getFiscalStorageNumber() : string|null

```php
public getFiscalStorageNumber() : string|null
```

**Summary**

Возвращает номер фискального накопителя в кассовом аппарате.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - Номер фискального накопителя в кассовом аппарате


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор чека в ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - Идентификатор чека в ЮKassa


<a name="method_getItems" class="anchor"></a>
#### public getItems() : \YooKassa\Request\Receipts\ReceiptResponseItemInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getItems() : \YooKassa\Request\Receipts\ReceiptResponseItemInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** \YooKassa\Request\Receipts\ReceiptResponseItemInterface[]|\YooKassa\Common\ListObjectInterface - 


<a name="method_getObjectId" class="anchor"></a>
#### public getObjectId() : string|null

```php
public getObjectId() : string|null
```

**Summary**

Возвращает идентификатор платежа или возврата, для которого был сформирован чек.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - 


<a name="method_getOnBehalfOf" class="anchor"></a>
#### public getOnBehalfOf() : string|null

```php
public getOnBehalfOf() : string|null
```

**Summary**

Возвращает идентификатор магазин

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - 


<a name="method_getPaymentId" class="anchor"></a>
#### public getPaymentId() : string|null

```php
public getPaymentId() : string|null
```

**Summary**

Возвращает идентификатор платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\PaymentReceiptResponse](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md)

**Returns:** string|null - Идентификатор платежа


<a name="method_getReceiptIndustryDetails" class="anchor"></a>
#### public getReceiptIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface

```php
public getReceiptIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface - Отраслевой реквизит чека


<a name="method_getReceiptOperationalDetails" class="anchor"></a>
#### public getReceiptOperationalDetails() : \YooKassa\Model\Receipt\OperationalDetails|null

```php
public getReceiptOperationalDetails() : \YooKassa\Model\Receipt\OperationalDetails|null
```

**Summary**

Возвращает операционный реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** \YooKassa\Model\Receipt\OperationalDetails|null - Операционный реквизит чека


<a name="method_getRegisteredAt" class="anchor"></a>
#### public getRegisteredAt() : \DateTime|null

```php
public getRegisteredAt() : \DateTime|null
```

**Summary**

Возвращает дату и время формирования чека в фискальном накопителе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** \DateTime|null - Дата и время формирования чека в фискальном накопителе


<a name="method_getSettlements" class="anchor"></a>
#### public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface

```php
public getSettlements() : \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает Массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** \YooKassa\Model\Receipt\SettlementInterface[]|\YooKassa\Common\ListObjectInterface - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус доставки данных для чека в онлайн-кассу.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - Статус доставки данных для чека в онлайн-кассу ("pending", "succeeded" или "canceled")


<a name="method_getTaxSystemCode" class="anchor"></a>
#### public getTaxSystemCode() : int|null

```php
public getTaxSystemCode() : int|null
```

**Summary**

Возвращает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** int|null - Код системы налогообложения. Число 1-6.


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип чека в онлайн-кассе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** string|null - Тип чека в онлайн-кассе: приход "payment" или возврат "refund"


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_notEmpty" class="anchor"></a>
#### public notEmpty() : bool

```php
public notEmpty() : bool
```

**Summary**

Проверяет есть ли в чеке хотя бы одна позиция.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

**Returns:** bool - True если чек не пуст, false если в чеке нет ни одной позиции


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setFiscalAttribute" class="anchor"></a>
#### public setFiscalAttribute() : self

```php
public setFiscalAttribute(string|null $fiscal_attribute = null) : self
```

**Summary**

Устанавливает фискальный признак чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | fiscal_attribute  | Фискальный признак чека. Формируется фискальным накопителем на основе данных, переданных для регистрации чека. |

**Returns:** self - 


<a name="method_setFiscalDocumentNumber" class="anchor"></a>
#### public setFiscalDocumentNumber() : self

```php
public setFiscalDocumentNumber(string|null $fiscal_document_number = null) : self
```

**Summary**

Устанавливает номер фискального документа

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | fiscal_document_number  | Номер фискального документа. |

**Returns:** self - 


<a name="method_setFiscalProviderId" class="anchor"></a>
#### public setFiscalProviderId() : self

```php
public setFiscalProviderId(string|null $fiscal_provider_id = null) : self
```

**Summary**

Устанавливает идентификатор чека в онлайн-кассе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | fiscal_provider_id  | Идентификатор чека в онлайн-кассе. Присутствует, если чек удалось зарегистрировать. |

**Returns:** self - 


<a name="method_setFiscalStorageNumber" class="anchor"></a>
#### public setFiscalStorageNumber() : self

```php
public setFiscalStorageNumber(string|null $fiscal_storage_number = null) : self
```

**Summary**

Устанавливает номер фискального накопителя в кассовом аппарате.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | fiscal_storage_number  | Номер фискального накопителя в кассовом аппарате. |

**Returns:** self - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string $id) : self
```

**Summary**

Устанавливает идентификатор чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | id  | Идентификатор чека |

**Returns:** self - 


<a name="method_setItems" class="anchor"></a>
#### public setItems() : self

```php
public setItems(\YooKassa\Request\Receipts\ReceiptResponseItemInterface[]|null $items) : self
```

**Summary**

Устанавливает список позиций в чеке.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Receipts\ReceiptResponseItemInterface[] OR null</code> | items  | Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров |

**Returns:** self - 


<a name="method_setObjectId" class="anchor"></a>
#### public setObjectId() : self

```php
public setObjectId(string|null $object_id) : self
```

**Summary**

Устанавливает идентификатор платежа или возврата, для которого был сформирован чек.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | object_id  |  |

**Returns:** self - 


<a name="method_setOnBehalfOf" class="anchor"></a>
#### public setOnBehalfOf() : self

```php
public setOnBehalfOf(string|null $on_behalf_of = null) : self
```

**Summary**

Возвращает идентификатор магазина, от имени которого нужно отправить чек.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | on_behalf_of  | Идентификатор магазина, от имени которого нужно отправить чек |

**Returns:** self - 


<a name="method_setPaymentId" class="anchor"></a>
#### public setPaymentId() : self

```php
public setPaymentId(string|null $payment_id = null) : self
```

**Summary**

Устанавливает идентификатор платежа в ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\PaymentReceiptResponse](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_id  | Идентификатор платежа в ЮKassa |

**Returns:** self - 


<a name="method_setReceiptIndustryDetails" class="anchor"></a>
#### public setReceiptIndustryDetails() : self

```php
public setReceiptIndustryDetails(array|\YooKassa\Model\Receipt\IndustryDetails[]|null $receipt_industry_details = null) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\IndustryDetails[] OR null</code> | receipt_industry_details  | Отраслевой реквизит чека |

**Returns:** self - 


<a name="method_setReceiptOperationalDetails" class="anchor"></a>
#### public setReceiptOperationalDetails() : self

```php
public setReceiptOperationalDetails(array|\YooKassa\Model\Receipt\OperationalDetails|null $receipt_operational_details = null) : self
```

**Summary**

Устанавливает операционный реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\OperationalDetails OR null</code> | receipt_operational_details  | Операционный реквизит чека |

**Returns:** self - 


<a name="method_setRegisteredAt" class="anchor"></a>
#### public setRegisteredAt() : self

```php
public setRegisteredAt(\DateTime|string|null $registered_at = null) : self
```

**Summary**

Устанавливает дату и время формирования чека в фискальном накопителе.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | registered_at  | Дата и время формирования чека в фискальном накопителе. Указывается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). |

**Returns:** self - 


<a name="method_setSettlements" class="anchor"></a>
#### public setSettlements() : self

```php
public setSettlements(\YooKassa\Model\Receipt\SettlementInterface[]|null $settlements) : self
```

**Summary**

Устанавливает массив оплат, обеспечивающих выдачу товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Receipt\SettlementInterface[] OR null</code> | settlements  |  |

**Returns:** self - 


<a name="method_setSpecificProperties" class="anchor"></a>
#### public setSpecificProperties() : void

```php
public setSpecificProperties(array $receiptData) : void
```

**Summary**

Установка свойств, присущих конкретному объекту.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\PaymentReceiptResponse](../classes/YooKassa-Request-Receipts-PaymentReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | receiptData  |  |

**Returns:** void - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status) : self
```

**Summary**

Устанавливает состояние регистрации фискального чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Состояние регистрации фискального чека |

**Returns:** self - 


<a name="method_setTaxSystemCode" class="anchor"></a>
#### public setTaxSystemCode() : self

```php
public setTaxSystemCode(int|null $tax_system_code) : self
```

**Summary**

Устанавливает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | tax_system_code  | Код системы налогообложения. Число 1-6 |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string $type) : self
```

**Summary**

Устанавливает типа чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\AbstractReceiptResponse](../classes/YooKassa-Request-Receipts-AbstractReceiptResponse.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  | Тип чека |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney