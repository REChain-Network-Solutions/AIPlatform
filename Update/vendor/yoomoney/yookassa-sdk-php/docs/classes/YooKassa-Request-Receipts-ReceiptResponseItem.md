# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Receipts\ReceiptResponseItem
### Namespace: [\YooKassa\Request\Receipts](../namespaces/yookassa-request-receipts.md)
---
**Summary:**

Класс, описывающий товар в чеке.


---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [COUNTRY_CODE_LENGTH](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#constant_COUNTRY_CODE_LENGTH) |  |  |
| public | [MAX_DECLARATION_NUMBER_LENGTH](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#constant_MAX_DECLARATION_NUMBER_LENGTH) |  |  |
| public | [MAX_PRODUCT_CODE_LENGTH](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#constant_MAX_PRODUCT_CODE_LENGTH) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$agent_type](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_agent_type) |  | Тип посредника, реализующего товар или услугу |
| public | [$agentType](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_agentType) |  | Тип посредника, реализующего товар или услугу |
| public | [$amount](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_amount) |  | Суммарная стоимость покупаемого товара в копейках/центах |
| public | [$country_of_origin_code](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_country_of_origin_code) |  | Код страны происхождения товара (тег в 54 ФЗ — 1230) |
| public | [$countryOfOriginCode](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_countryOfOriginCode) |  | Код страны происхождения товара (тег в 54 ФЗ — 1230) |
| public | [$customs_declaration_number](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_customs_declaration_number) |  | Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231 |
| public | [$customsDeclarationNumber](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_customsDeclarationNumber) |  | Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231 |
| public | [$description](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_description) |  | Наименование товара (тег в 54 ФЗ — 1030) |
| public | [$excise](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_excise) |  | Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229) |
| public | [$mark_code_info](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_mark_code_info) |  | Код товара (тег в 54 ФЗ — 1163) |
| public | [$mark_mode](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_mark_mode) |  | Режим обработки кода маркировки (тег в 54 ФЗ — 2102) |
| public | [$mark_quantity](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_mark_quantity) |  | Дробное количество маркированного товара (тег в 54 ФЗ — 1291) |
| public | [$markCodeInfo](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_markCodeInfo) |  | Код товара (тег в 54 ФЗ — 1163) |
| public | [$markMode](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_markMode) |  | Режим обработки кода маркировки (тег в 54 ФЗ — 2102) |
| public | [$markQuantity](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_markQuantity) |  | Дробное количество маркированного товара (тег в 54 ФЗ — 1291) |
| public | [$measure](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_measure) |  | Мера количества предмета расчета (тег в 54 ФЗ — 2108) |
| public | [$payment_mode](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_payment_mode) |  | Признак способа расчета (тег в 54 ФЗ — 1214) |
| public | [$payment_subject](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_payment_subject) |  | Признак предмета расчета (тег в 54 ФЗ — 1212) |
| public | [$payment_subject_industry_details](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_payment_subject_industry_details) |  | Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260) |
| public | [$paymentMode](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_paymentMode) |  | Признак способа расчета (тег в 54 ФЗ — 1214) |
| public | [$paymentSubject](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_paymentSubject) |  | Признак предмета расчета (тег в 54 ФЗ — 1212) |
| public | [$paymentSubjectIndustryDetails](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_paymentSubjectIndustryDetails) |  | Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260) |
| public | [$price](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_price) |  | Цена товара (тег в 54 ФЗ — 1079) |
| public | [$product_code](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_product_code) |  | Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162) |
| public | [$productCode](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_productCode) |  | Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162) |
| public | [$quantity](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_quantity) |  | Количество (тег в 54 ФЗ — 1023) |
| public | [$supplier](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_supplier) |  | Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224) |
| public | [$vat_code](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_vat_code) |  | Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199) |
| public | [$vatCode](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#property_vatCode) |  | Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199) |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAgentType()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getAgentType) |  | Возвращает тип посредника, реализующего товар или услугу. |
| public | [getAmount()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getAmount) |  | Возвращает общую стоимость покупаемого товара в копейках/центах. |
| public | [getCountryOfOriginCode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getCountryOfOriginCode) |  | Возвращает код страны происхождения товара по общероссийскому классификатору стран мира. |
| public | [getCustomsDeclarationNumber()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getCustomsDeclarationNumber) |  | Возвращает номер таможенной декларации. |
| public | [getDescription()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getDescription) |  | Возвращает наименование товара. |
| public | [getExcise()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getExcise) |  | Возвращает сумму акциза товара с учетом копеек. |
| public | [getMarkCodeInfo()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getMarkCodeInfo) |  | Возвращает код товара. |
| public | [getMarkMode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getMarkMode) |  | Возвращает режим обработки кода маркировки. |
| public | [getMarkQuantity()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getMarkQuantity) |  | Возвращает дробное количество маркированного товара. |
| public | [getMeasure()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getMeasure) |  | Возвращает меру количества предмета расчета. |
| public | [getPaymentMode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getPaymentMode) |  | Возвращает признак способа расчета. |
| public | [getPaymentSubject()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getPaymentSubject) |  | Возвращает признак предмета расчета. |
| public | [getPaymentSubjectIndustryDetails()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getPaymentSubjectIndustryDetails) |  | Возвращает отраслевой реквизит чека. |
| public | [getPrice()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getPrice) |  | Возвращает цену товара. |
| public | [getProductCode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getProductCode) |  | Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке. |
| public | [getQuantity()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getQuantity) |  | Возвращает количество товара. |
| public | [getSupplier()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getSupplier) |  | Возвращает информацию о поставщике товара или услуги |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [getVatCode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_getVatCode) |  | Возвращает ставку НДС |
| public | [jsonSerialize()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAgentType()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setAgentType) |  | Устанавливает тип посредника, реализующего товар или услугу. |
| public | [setCountryOfOriginCode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setCountryOfOriginCode) |  | Устанавливает код страны происхождения товара по общероссийскому классификатору стран мира. |
| public | [setCustomsDeclarationNumber()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setCustomsDeclarationNumber) |  | Устанавливает номер таможенной декларации (от 1 до 32 символов). |
| public | [setDescription()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setDescription) |  | Устанавливает наименование товара. |
| public | [setExcise()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setExcise) |  | Устанавливает сумму акциза товара с учетом копеек. |
| public | [setMarkCodeInfo()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setMarkCodeInfo) |  | Устанавливает код товара. |
| public | [setMarkMode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setMarkMode) |  | Устанавливает режим обработки кода маркировки. |
| public | [setMarkQuantity()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setMarkQuantity) |  | Устанавливает дробное количество маркированного товара. |
| public | [setMeasure()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setMeasure) |  | Устанавливает меру количества предмета расчета. |
| public | [setPaymentMode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setPaymentMode) |  | Устанавливает признак способа расчета. |
| public | [setPaymentSubject()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setPaymentSubject) |  | Устанавливает признак предмета расчета. |
| public | [setPaymentSubjectIndustryDetails()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setPaymentSubjectIndustryDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setPrice()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setPrice) |  | Устанавливает цену товара. |
| public | [setProductCode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setProductCode) |  | Устанавливает код товара — уникальный номер, который присваивается экземпляру товара при маркировке. |
| public | [setQuantity()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setQuantity) |  | Устанавливает количество покупаемого товара. |
| public | [setSupplier()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setSupplier) |  | Устанавливает информацию о поставщике товара или услуги. |
| public | [setVatCode()](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md#method_setVatCode) |  | Устанавливает ставку НДС |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Receipts/ReceiptResponseItem.php](../../lib/Request/Receipts/ReceiptResponseItem.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Receipts\ReceiptResponseItem
* Implements:
  * [\YooKassa\Request\Receipts\ReceiptResponseItemInterface](../classes/YooKassa-Request-Receipts-ReceiptResponseItemInterface.md)

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
<a name="constant_COUNTRY_CODE_LENGTH" class="anchor"></a>
###### COUNTRY_CODE_LENGTH
```php
COUNTRY_CODE_LENGTH = 2 : int
```


<a name="constant_MAX_DECLARATION_NUMBER_LENGTH" class="anchor"></a>
###### MAX_DECLARATION_NUMBER_LENGTH
```php
MAX_DECLARATION_NUMBER_LENGTH = 32 : int
```


<a name="constant_MAX_PRODUCT_CODE_LENGTH" class="anchor"></a>
###### MAX_PRODUCT_CODE_LENGTH
```php
MAX_PRODUCT_CODE_LENGTH = 96 : int
```



---
## Properties
<a name="property_agent_type"></a>
#### public $agent_type : string
---
***Description***

Тип посредника, реализующего товар или услугу

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_agentType"></a>
#### public $agentType : string
---
***Description***

Тип посредника, реализующего товар или услугу

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_amount"></a>
#### public $amount : float
---
***Description***

Суммарная стоимость покупаемого товара в копейках/центах

**Type:** <a href="../float"><abbr title="float">float</abbr></a>

**Details:**


<a name="property_country_of_origin_code"></a>
#### public $country_of_origin_code : string
---
***Description***

Код страны происхождения товара (тег в 54 ФЗ — 1230)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_countryOfOriginCode"></a>
#### public $countryOfOriginCode : string
---
***Description***

Код страны происхождения товара (тег в 54 ФЗ — 1230)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_customs_declaration_number"></a>
#### public $customs_declaration_number : string
---
***Description***

Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_customsDeclarationNumber"></a>
#### public $customsDeclarationNumber : string
---
***Description***

Номер таможенной декларации (от 1 до 32 символов). Тег в 54 ФЗ — 1231

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Наименование товара (тег в 54 ФЗ — 1030)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_excise"></a>
#### public $excise : float
---
***Description***

Сумма акциза товара с учетом копеек (тег в 54 ФЗ — 1229)

**Type:** <a href="../float"><abbr title="float">float</abbr></a>

**Details:**


<a name="property_mark_code_info"></a>
#### public $mark_code_info : \YooKassa\Model\Receipt\MarkCodeInfo
---
***Description***

Код товара (тег в 54 ФЗ — 1163)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkCodeInfo.html"><abbr title="\YooKassa\Model\Receipt\MarkCodeInfo">MarkCodeInfo</abbr></a>

**Details:**


<a name="property_mark_mode"></a>
#### public $mark_mode : string
---
***Description***

Режим обработки кода маркировки (тег в 54 ФЗ — 2102)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_mark_quantity"></a>
#### public $mark_quantity : \YooKassa\Model\Receipt\MarkQuantity
---
***Description***

Дробное количество маркированного товара (тег в 54 ФЗ — 1291)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkQuantity.html"><abbr title="\YooKassa\Model\Receipt\MarkQuantity">MarkQuantity</abbr></a>

**Details:**


<a name="property_markCodeInfo"></a>
#### public $markCodeInfo : \YooKassa\Model\Receipt\MarkCodeInfo
---
***Description***

Код товара (тег в 54 ФЗ — 1163)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkCodeInfo.html"><abbr title="\YooKassa\Model\Receipt\MarkCodeInfo">MarkCodeInfo</abbr></a>

**Details:**


<a name="property_markMode"></a>
#### public $markMode : string
---
***Description***

Режим обработки кода маркировки (тег в 54 ФЗ — 2102)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_markQuantity"></a>
#### public $markQuantity : \YooKassa\Model\Receipt\MarkQuantity
---
***Description***

Дробное количество маркированного товара (тег в 54 ФЗ — 1291)

**Type:** <a href="../classes/YooKassa-Model-Receipt-MarkQuantity.html"><abbr title="\YooKassa\Model\Receipt\MarkQuantity">MarkQuantity</abbr></a>

**Details:**


<a name="property_measure"></a>
#### public $measure : string
---
***Description***

Мера количества предмета расчета (тег в 54 ФЗ — 2108)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_mode"></a>
#### public $payment_mode : string
---
***Description***

Признак способа расчета (тег в 54 ФЗ — 1214)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_subject"></a>
#### public $payment_subject : string
---
***Description***

Признак предмета расчета (тег в 54 ФЗ — 1212)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_subject_industry_details"></a>
#### public $payment_subject_industry_details : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**


<a name="property_paymentMode"></a>
#### public $paymentMode : string
---
***Description***

Признак способа расчета (тег в 54 ФЗ — 1214)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_paymentSubject"></a>
#### public $paymentSubject : string
---
***Description***

Признак предмета расчета (тег в 54 ФЗ — 1212)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_paymentSubjectIndustryDetails"></a>
#### public $paymentSubjectIndustryDetails : \YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]
---
***Description***

Отраслевой реквизит предмета расчета (тег в 54 ФЗ — 1260)

**Type:** <a href="../\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]"><abbr title="\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Receipt\IndustryDetails[]">IndustryDetails[]</abbr></a>

**Details:**


<a name="property_price"></a>
#### public $price : \YooKassa\Model\AmountInterface
---
***Description***

Цена товара (тег в 54 ФЗ — 1079)

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_product_code"></a>
#### public $product_code : string
---
***Description***

Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_productCode"></a>
#### public $productCode : string
---
***Description***

Код товара — уникальный номер, который присваивается экземпляру товара при маркировке (тег в 54 ФЗ — 1162)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_quantity"></a>
#### public $quantity : float
---
***Description***

Количество (тег в 54 ФЗ — 1023)

**Type:** <a href="../float"><abbr title="float">float</abbr></a>

**Details:**


<a name="property_supplier"></a>
#### public $supplier : \YooKassa\Model\Receipt\Supplier
---
***Description***

Информация о поставщике товара или услуги (тег в 54 ФЗ — 1224)

**Type:** <a href="../classes/YooKassa-Model-Receipt-Supplier.html"><abbr title="\YooKassa\Model\Receipt\Supplier">Supplier</abbr></a>

**Details:**


<a name="property_vat_code"></a>
#### public $vat_code : int
---
***Description***

Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199)

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_vatCode"></a>
#### public $vatCode : int
---
***Description***

Ставка НДС, число 1-10 (тег в 54 ФЗ — 1199)

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**



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


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | sourceArray  | Массив с информацией о товаре, пришедший от API |

**Returns:** void - 


<a name="method_getAgentType" class="anchor"></a>
#### public getAgentType() : string|null

```php
public getAgentType() : string|null
```

**Summary**

Возвращает тип посредника, реализующего товар или услугу.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** string|null - Тип посредника


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : int|null

```php
public getAmount() : int|null
```

**Summary**

Возвращает общую стоимость покупаемого товара в копейках/центах.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** int|null - Сумма стоимости покупаемого товара


<a name="method_getCountryOfOriginCode" class="anchor"></a>
#### public getCountryOfOriginCode() : null|string

```php
public getCountryOfOriginCode() : null|string
```

**Summary**

Возвращает код страны происхождения товара по общероссийскому классификатору стран мира.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** null|string - Код страны происхождения товара


<a name="method_getCustomsDeclarationNumber" class="anchor"></a>
#### public getCustomsDeclarationNumber() : null|string

```php
public getCustomsDeclarationNumber() : null|string
```

**Summary**

Возвращает номер таможенной декларации.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** null|string - Номер таможенной декларации (от 1 до 32 символов)


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает наименование товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** string|null - Наименование товара


<a name="method_getExcise" class="anchor"></a>
#### public getExcise() : null|float

```php
public getExcise() : null|float
```

**Summary**

Возвращает сумму акциза товара с учетом копеек.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** null|float - Сумма акциза товара с учетом копеек


<a name="method_getMarkCodeInfo" class="anchor"></a>
#### public getMarkCodeInfo() : \YooKassa\Model\Receipt\MarkCodeInfo|null

```php
public getMarkCodeInfo() : \YooKassa\Model\Receipt\MarkCodeInfo|null
```

**Summary**

Возвращает код товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** \YooKassa\Model\Receipt\MarkCodeInfo|null - Код товара


<a name="method_getMarkMode" class="anchor"></a>
#### public getMarkMode() : string|null

```php
public getMarkMode() : string|null
```

**Summary**

Возвращает режим обработки кода маркировки.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** string|null - Режим обработки кода маркировки


<a name="method_getMarkQuantity" class="anchor"></a>
#### public getMarkQuantity() : \YooKassa\Model\Receipt\MarkQuantity|null

```php
public getMarkQuantity() : \YooKassa\Model\Receipt\MarkQuantity|null
```

**Summary**

Возвращает дробное количество маркированного товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** \YooKassa\Model\Receipt\MarkQuantity|null - Дробное количество маркированного товара


<a name="method_getMeasure" class="anchor"></a>
#### public getMeasure() : string|null

```php
public getMeasure() : string|null
```

**Summary**

Возвращает меру количества предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** string|null - Мера количества предмета расчета


<a name="method_getPaymentMode" class="anchor"></a>
#### public getPaymentMode() : null|string

```php
public getPaymentMode() : null|string
```

**Summary**

Возвращает признак способа расчета.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** null|string - Признак способа расчета


<a name="method_getPaymentSubject" class="anchor"></a>
#### public getPaymentSubject() : null|string

```php
public getPaymentSubject() : null|string
```

**Summary**

Возвращает признак предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** null|string - Признак предмета расчета


<a name="method_getPaymentSubjectIndustryDetails" class="anchor"></a>
#### public getPaymentSubjectIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface

```php
public getPaymentSubjectIndustryDetails() : \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** \YooKassa\Model\Receipt\IndustryDetails[]|\YooKassa\Common\ListObjectInterface - Отраслевой реквизит чека


<a name="method_getPrice" class="anchor"></a>
#### public getPrice() : \YooKassa\Model\AmountInterface|null

```php
public getPrice() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает цену товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Цена товара


<a name="method_getProductCode" class="anchor"></a>
#### public getProductCode() : null|string

```php
public getProductCode() : null|string
```

**Summary**

Возвращает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** null|string - Код товара


<a name="method_getQuantity" class="anchor"></a>
#### public getQuantity() : float

```php
public getQuantity() : float
```

**Summary**

Возвращает количество товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** float - Количество купленного товара


<a name="method_getSupplier" class="anchor"></a>
#### public getSupplier() : \YooKassa\Model\Receipt\SupplierInterface|null

```php
public getSupplier() : \YooKassa\Model\Receipt\SupplierInterface|null
```

**Summary**

Возвращает информацию о поставщике товара или услуги

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** \YooKassa\Model\Receipt\SupplierInterface|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_getVatCode" class="anchor"></a>
#### public getVatCode() : null|int

```php
public getVatCode() : null|int
```

**Summary**

Возвращает ставку НДС

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** null|int - Ставка НДС, число 1-10, или null, если ставка не задана


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


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


<a name="method_setAgentType" class="anchor"></a>
#### public setAgentType() : self

```php
public setAgentType(string|null $agent_type = null) : self
```

**Summary**

Устанавливает тип посредника, реализующего товар или услугу.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | agent_type  | Тип посредника |

**Returns:** self - 


<a name="method_setCountryOfOriginCode" class="anchor"></a>
#### public setCountryOfOriginCode() : self

```php
public setCountryOfOriginCode(string|null $country_of_origin_code = null) : self
```

**Summary**

Устанавливает код страны происхождения товара по общероссийскому классификатору стран мира.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | country_of_origin_code  | Код страны происхождения товара |

**Returns:** self - 


<a name="method_setCustomsDeclarationNumber" class="anchor"></a>
#### public setCustomsDeclarationNumber() : self

```php
public setCustomsDeclarationNumber(string|null $customs_declaration_number = null) : self
```

**Summary**

Устанавливает номер таможенной декларации (от 1 до 32 символов).

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | customs_declaration_number  | Номер таможенной декларации |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description) : self
```

**Summary**

Устанавливает наименование товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Наименование товара |

**Returns:** self - 


<a name="method_setExcise" class="anchor"></a>
#### public setExcise() : self

```php
public setExcise(float|null $excise = null) : self
```

**Summary**

Устанавливает сумму акциза товара с учетом копеек.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | excise  | Сумма акциза товара с учетом копеек |

**Returns:** self - 


<a name="method_setMarkCodeInfo" class="anchor"></a>
#### public setMarkCodeInfo() : self

```php
public setMarkCodeInfo(array|\YooKassa\Model\Receipt\MarkCodeInfo|null $mark_code_info = null) : self
```

**Summary**

Устанавливает код товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\MarkCodeInfo OR null</code> | mark_code_info  | Код товара |

**Returns:** self - 


<a name="method_setMarkMode" class="anchor"></a>
#### public setMarkMode() : self

```php
public setMarkMode(string|null $mark_mode = null) : self
```

**Summary**

Устанавливает режим обработки кода маркировки.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | mark_mode  | Режим обработки кода маркировки |

**Returns:** self - 


<a name="method_setMarkQuantity" class="anchor"></a>
#### public setMarkQuantity() : self

```php
public setMarkQuantity(array|\YooKassa\Model\Receipt\MarkQuantity|null $mark_quantity = null) : self
```

**Summary**

Устанавливает дробное количество маркированного товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\MarkQuantity OR null</code> | mark_quantity  | Дробное количество маркированного товара |

**Returns:** self - 


<a name="method_setMeasure" class="anchor"></a>
#### public setMeasure() : self

```php
public setMeasure(string|null $measure) : self
```

**Summary**

Устанавливает меру количества предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | measure  | Мера количества предмета расчета |

**Returns:** self - 


<a name="method_setPaymentMode" class="anchor"></a>
#### public setPaymentMode() : self

```php
public setPaymentMode(string|null $payment_mode = null) : self
```

**Summary**

Устанавливает признак способа расчета.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_mode  | Признак способа расчета |

**Returns:** self - 


<a name="method_setPaymentSubject" class="anchor"></a>
#### public setPaymentSubject() : self

```php
public setPaymentSubject(null|string $payment_subject = null) : self
```

**Summary**

Устанавливает признак предмета расчета.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | payment_subject  | Признак предмета расчета |

**Returns:** self - 


<a name="method_setPaymentSubjectIndustryDetails" class="anchor"></a>
#### public setPaymentSubjectIndustryDetails() : self

```php
public setPaymentSubjectIndustryDetails(array|\YooKassa\Model\Receipt\IndustryDetails[]|null $payment_subject_industry_details = null) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\IndustryDetails[] OR null</code> | payment_subject_industry_details  | Отраслевой реквизит чека |

**Returns:** self - 


<a name="method_setPrice" class="anchor"></a>
#### public setPrice() : void

```php
public setPrice(\YooKassa\Model\AmountInterface|array|null $value) : void
```

**Summary**

Устанавливает цену товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | value  | Цена товара |

**Returns:** void - 


<a name="method_setProductCode" class="anchor"></a>
#### public setProductCode() : self

```php
public setProductCode(\YooKassa\Helpers\ProductCode|string $product_code) : self
```

**Summary**

Устанавливает код товара — уникальный номер, который присваивается экземпляру товара при маркировке.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Helpers\ProductCode OR string</code> | product_code  | Код товара |

**Returns:** self - 


<a name="method_setQuantity" class="anchor"></a>
#### public setQuantity() : self

```php
public setQuantity(float|null $quantity) : self
```

**Summary**

Устанавливает количество покупаемого товара.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">float OR null</code> | quantity  | Количество |

**Returns:** self - 


<a name="method_setSupplier" class="anchor"></a>
#### public setSupplier() : self

```php
public setSupplier(array|\YooKassa\Model\Receipt\SupplierInterface|null $supplier = null) : self
```

**Summary**

Устанавливает информацию о поставщике товара или услуги.

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\SupplierInterface OR null</code> | supplier  | Информация о поставщике товара или услуги |

**Returns:** self - 


<a name="method_setVatCode" class="anchor"></a>
#### public setVatCode() : self

```php
public setVatCode(int|null $vat_code = null) : self
```

**Summary**

Устанавливает ставку НДС

**Details:**
* Inherited From: [\YooKassa\Request\Receipts\ReceiptResponseItem](../classes/YooKassa-Request-Receipts-ReceiptResponseItem.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | vat_code  | Ставка НДС, число 1-10 |

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