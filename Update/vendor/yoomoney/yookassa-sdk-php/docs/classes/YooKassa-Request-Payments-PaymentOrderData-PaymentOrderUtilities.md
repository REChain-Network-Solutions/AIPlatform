# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities
### Namespace: [\YooKassa\Request\Payments\PaymentOrderData](../namespaces/yookassa-request-payments-paymentorderdata.md)
---
**Summary:**

Класс, представляющий модель PaymentOrderUtilities.

**Description:**

Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ.

Необходимо передавать при [оплате ЖКУ](/developers/payment-acceptance/scenario-extensions/utility-payments).

Кроме параметров, отмеченных как обязательные, должен быть передан как минимум один параметр из этого списка: `payment_document_id`, `payment_document_number`, `account_number`, `unified_account_number` или `service_id`.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$account_number](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_account_number) |  | Номер лицевого счета на стороне поставщика ЖКУ.  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `unified_account_number` или `service_id`. |
| public | [$amount](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_amount) |  | Сумма платежного поручения — сумма, которую пользователь переводит получателю платежа. Равна общей сумме платежа. |
| public | [$kbk](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_kbk) |  | Код бюджетной классификации (КБК). |
| public | [$oktmo](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_oktmo) |  | Код ОКТМО (Общероссийский классификатор территорий муниципальных образований). |
| public | [$payment_document_id](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_payment_document_id) |  | Идентификатор платежного документа.  Обязательный параметр, если не передан `payment_document_number`, `account_number`, `unified_account_number` или `service_id`. |
| public | [$payment_document_number](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_payment_document_number) |  | Номер платежного документа на стороне поставщика ЖКУ.  Обязательный параметр, если не передан `payment_document_id`, `account_number`, `unified_account_number` или `service_id`. |
| public | [$payment_period](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_payment_period) |  | Период оплаты, за который выставлены начисления и за который вносится оплата. |
| public | [$payment_purpose](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_payment_purpose) |  | Назначение платежа (не больше 210 символов). |
| public | [$recipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_recipient) |  | Получатель платежа — государственная или коммерческая организация, которая предоставляет услуги или является информационным посредником, который собирает и обрабатывает начисления от других поставщиков услуг. |
| public | [$service_id](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_service_id) |  | Идентификатор жилищно-коммунальной услуги (ЖКУ).  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `unified_account_number`. |
| public | [$type](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_type) |  | Код вида платежного поручения. |
| public | [$type](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md#property_type) |  | Код вида платежного поручения |
| public | [$unified_account_number](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#property_unified_account_number) |  | Единый лицевой счет. Уникальный идентификатор в ГИС ЖКХ, который характеризует связку «собственник-помещение».  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `service_id`. |
| protected | [$_type](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md#property__type) |  | Код вида платежного поручения |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method___construct) |  |  |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAccountNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getAccountNumber) |  | Возвращает номер лицевого счета на стороне поставщика ЖКУ. |
| public | [getAmount()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getAmount) |  | Возвращает amount. |
| public | [getKbk()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getKbk) |  | Возвращает kbk. |
| public | [getOktmo()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getOktmo) |  | Возвращает oktmo. |
| public | [getPaymentDocumentId()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getPaymentDocumentId) |  | Возвращает payment_document_id. |
| public | [getPaymentDocumentNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getPaymentDocumentNumber) |  | Возвращает payment_document_number. |
| public | [getPaymentPeriod()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getPaymentPeriod) |  | Возвращает payment_period. |
| public | [getPaymentPurpose()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getPaymentPurpose) |  | Возвращает payment_purpose. |
| public | [getRecipient()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getRecipient) |  | Возвращает recipient. |
| public | [getServiceId()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getServiceId) |  | Возвращает идентификатор жилищно-коммунальной услуги (ЖКУ). |
| public | [getType()](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md#method_getType) |  | Возвращает type. |
| public | [getUnifiedAccountNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_getUnifiedAccountNumber) |  | Возвращает единый лицевой счет. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasAccountNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_hasAccountNumber) |  | Проверяет, были ли установлен номер лицевого счета на стороне поставщика ЖКУ. |
| public | [hasPaymentDocumentId()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_hasPaymentDocumentId) |  | Проверяет, были ли установлен идентификатор платежного документа. |
| public | [hasPaymentDocumentNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_hasPaymentDocumentNumber) |  | Проверяет, были ли установлен номер платежного документа на стороне поставщика ЖКУ. |
| public | [hasServiceId()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_hasServiceId) |  | Проверяет, были ли установлен идентификатор жилищно-коммунальной услуги (ЖКУ). |
| public | [hasUnifiedAccountNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_hasUnifiedAccountNumber) |  | Проверяет, были ли установлен единый лицевой счет. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAccountNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setAccountNumber) |  | Устанавливает номер лицевого счета на стороне поставщика ЖКУ. |
| public | [setAmount()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setAmount) |  | Устанавливает amount. |
| public | [setKbk()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setKbk) |  | Устанавливает kbk. |
| public | [setOktmo()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setOktmo) |  | Устанавливает oktmo. |
| public | [setPaymentDocumentId()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setPaymentDocumentId) |  | Устанавливает идентификатор платежного документа. |
| public | [setPaymentDocumentNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setPaymentDocumentNumber) |  | Устанавливает номер платежного документа на стороне поставщика ЖКУ. |
| public | [setPaymentPeriod()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setPaymentPeriod) |  | Устанавливает payment_period. |
| public | [setPaymentPurpose()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setPaymentPurpose) |  | Устанавливает payment_purpose. |
| public | [setRecipient()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setRecipient) |  | Устанавливает recipient. |
| public | [setServiceId()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setServiceId) |  | Устанавливает идентификатор жилищно-коммунальной услуги (ЖКУ). |
| public | [setType()](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md#method_setType) |  | Устанавливает type. |
| public | [setUnifiedAccountNumber()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_setUnifiedAccountNumber) |  | Устанавливает единый лицевой счет. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md#method_validate) |  | Проверяет, что все необходимые поля заданы. |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/PaymentOrderData/PaymentOrderUtilities.php](../../lib/Request/Payments/PaymentOrderData/PaymentOrderUtilities.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md)
  * \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_account_number"></a>
#### public $account_number : string
---
***Description***

Номер лицевого счета на стороне поставщика ЖКУ.  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `unified_account_number` или `service_id`.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\MonetaryAmount
---
***Description***

Сумма платежного поручения — сумма, которую пользователь переводит получателю платежа. Равна общей сумме платежа.

**Type:** <a href="../classes/YooKassa-Model-MonetaryAmount.html"><abbr title="\YooKassa\Model\MonetaryAmount">MonetaryAmount</abbr></a>

**Details:**


<a name="property_kbk"></a>
#### public $kbk : string
---
***Description***

Код бюджетной классификации (КБК).

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_oktmo"></a>
#### public $oktmo : string
---
***Description***

Код ОКТМО (Общероссийский классификатор территорий муниципальных образований).

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_document_id"></a>
#### public $payment_document_id : string
---
***Description***

Идентификатор платежного документа.  Обязательный параметр, если не передан `payment_document_number`, `account_number`, `unified_account_number` или `service_id`.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_document_number"></a>
#### public $payment_document_number : string
---
***Description***

Номер платежного документа на стороне поставщика ЖКУ.  Обязательный параметр, если не передан `payment_document_id`, `account_number`, `unified_account_number` или `service_id`.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_payment_period"></a>
#### public $payment_period : \YooKassa\Request\Payments\PaymentPeriod
---
***Description***

Период оплаты, за который выставлены начисления и за который вносится оплата.

**Type:** <a href="../classes/YooKassa-Request-Payments-PaymentPeriod.html"><abbr title="\YooKassa\Request\Payments\PaymentPeriod">PaymentPeriod</abbr></a>

**Details:**


<a name="property_payment_purpose"></a>
#### public $payment_purpose : string
---
***Description***

Назначение платежа (не больше 210 символов).

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_recipient"></a>
#### public $recipient : \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient
---
***Description***

Получатель платежа — государственная или коммерческая организация, которая предоставляет услуги или является информационным посредником, который собирает и обрабатывает начисления от других поставщиков услуг.

**Type:** <a href="../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.html"><abbr title="\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient">PaymentOrderRecipient</abbr></a>

**Details:**


<a name="property_service_id"></a>
#### public $service_id : string
---
***Description***

Идентификатор жилищно-коммунальной услуги (ЖКУ).  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `unified_account_number`.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Код вида платежного поручения.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Код вида платежного поручения

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md)


<a name="property_unified_account_number"></a>
#### public $unified_account_number : string
---
***Description***

Единый лицевой счет. Уникальный идентификатор в ГИС ЖКХ, который характеризует связку «собственник-помещение».  Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `service_id`.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property__type"></a>
#### protected $_type : ?string
---
**Summary**

Код вида платежного поручения

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(?array $data = []) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?array</code> | data  |  |

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


<a name="method_getAccountNumber" class="anchor"></a>
#### public getAccountNumber() : string|null

```php
public getAccountNumber() : string|null
```

**Summary**

Возвращает номер лицевого счета на стороне поставщика ЖКУ.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает amount.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** \YooKassa\Model\AmountInterface|null - 


<a name="method_getKbk" class="anchor"></a>
#### public getKbk() : string|null

```php
public getKbk() : string|null
```

**Summary**

Возвращает kbk.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getOktmo" class="anchor"></a>
#### public getOktmo() : string|null

```php
public getOktmo() : string|null
```

**Summary**

Возвращает oktmo.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getPaymentDocumentId" class="anchor"></a>
#### public getPaymentDocumentId() : string|null

```php
public getPaymentDocumentId() : string|null
```

**Summary**

Возвращает payment_document_id.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getPaymentDocumentNumber" class="anchor"></a>
#### public getPaymentDocumentNumber() : string|null

```php
public getPaymentDocumentNumber() : string|null
```

**Summary**

Возвращает payment_document_number.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getPaymentPeriod" class="anchor"></a>
#### public getPaymentPeriod() : \YooKassa\Request\Payments\PaymentPeriod|null

```php
public getPaymentPeriod() : \YooKassa\Request\Payments\PaymentPeriod|null
```

**Summary**

Возвращает payment_period.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** \YooKassa\Request\Payments\PaymentPeriod|null - 


<a name="method_getPaymentPurpose" class="anchor"></a>
#### public getPaymentPurpose() : string|null

```php
public getPaymentPurpose() : string|null
```

**Summary**

Возвращает payment_purpose.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getRecipient" class="anchor"></a>
#### public getRecipient() : \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient|null

```php
public getRecipient() : \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient|null
```

**Summary**

Возвращает recipient.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient|null - 


<a name="method_getServiceId" class="anchor"></a>
#### public getServiceId() : string|null

```php
public getServiceId() : string|null
```

**Summary**

Возвращает идентификатор жилищно-коммунальной услуги (ЖКУ).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает type.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md)

**Returns:** string|null - 


<a name="method_getUnifiedAccountNumber" class="anchor"></a>
#### public getUnifiedAccountNumber() : string|null

```php
public getUnifiedAccountNumber() : string|null
```

**Summary**

Возвращает единый лицевой счет.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** string|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasAccountNumber" class="anchor"></a>
#### public hasAccountNumber() : bool

```php
public hasAccountNumber() : bool
```

**Summary**

Проверяет, были ли установлен номер лицевого счета на стороне поставщика ЖКУ.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** bool - True если номер лицевого счета на стороне поставщика ЖКУ был установлен, false если нет


<a name="method_hasPaymentDocumentId" class="anchor"></a>
#### public hasPaymentDocumentId() : bool

```php
public hasPaymentDocumentId() : bool
```

**Summary**

Проверяет, были ли установлен идентификатор платежного документа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** bool - True если идентификатор платежного документа был установлен, false если нет


<a name="method_hasPaymentDocumentNumber" class="anchor"></a>
#### public hasPaymentDocumentNumber() : bool

```php
public hasPaymentDocumentNumber() : bool
```

**Summary**

Проверяет, были ли установлен номер платежного документа на стороне поставщика ЖКУ.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** bool - True если номер платежного документа на стороне поставщика ЖКУ был установлен, false если нет


<a name="method_hasServiceId" class="anchor"></a>
#### public hasServiceId() : bool

```php
public hasServiceId() : bool
```

**Summary**

Проверяет, были ли установлен идентификатор жилищно-коммунальной услуги (ЖКУ).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** bool - True если идентификатор жилищно-коммунальной услуги (ЖКУ) был установлен, false если нет


<a name="method_hasUnifiedAccountNumber" class="anchor"></a>
#### public hasUnifiedAccountNumber() : bool

```php
public hasUnifiedAccountNumber() : bool
```

**Summary**

Проверяет, были ли установлен единый лицевой счет.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** bool - True если единый лицевой счет был установлен, false если нет


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


<a name="method_setAccountNumber" class="anchor"></a>
#### public setAccountNumber() : self

```php
public setAccountNumber(string|null $account_number = null) : self
```

**Summary**

Устанавливает номер лицевого счета на стороне поставщика ЖКУ.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | account_number  | Номер лицевого счета на стороне поставщика ЖКУ. Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `unified_account_number` или `service_id`. |

**Returns:** self - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $amount = null) : self
```

**Summary**

Устанавливает amount.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | amount  | Сумма платежного поручения — сумма, которую пользователь переводит получателю платежа. Равна общей сумме платежа. |

**Returns:** self - 


<a name="method_setKbk" class="anchor"></a>
#### public setKbk() : self

```php
public setKbk(string|null $kbk = null) : self
```

**Summary**

Устанавливает kbk.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | kbk  | Код бюджетной классификации (КБК). |

**Returns:** self - 


<a name="method_setOktmo" class="anchor"></a>
#### public setOktmo() : self

```php
public setOktmo(string|null $oktmo = null) : self
```

**Summary**

Устанавливает oktmo.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | oktmo  | Код ОКТМО (Общероссийский классификатор территорий муниципальных образований). |

**Returns:** self - 


<a name="method_setPaymentDocumentId" class="anchor"></a>
#### public setPaymentDocumentId() : self

```php
public setPaymentDocumentId(string|null $payment_document_id = null) : self
```

**Summary**

Устанавливает идентификатор платежного документа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_document_id  | Идентификатор платежного документа. Обязательный параметр, если не передан `payment_document_number`, `account_number`, `unified_account_number` или `service_id`. |

**Returns:** self - 


<a name="method_setPaymentDocumentNumber" class="anchor"></a>
#### public setPaymentDocumentNumber() : self

```php
public setPaymentDocumentNumber(string|null $payment_document_number = null) : self
```

**Summary**

Устанавливает номер платежного документа на стороне поставщика ЖКУ.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payment_document_number  | Номер платежного документа на стороне поставщика ЖКУ. Обязательный параметр, если не передан `payment_document_id`, `account_number`, `unified_account_number` или `service_id`. |

**Returns:** self - 


<a name="method_setPaymentPeriod" class="anchor"></a>
#### public setPaymentPeriod() : self

```php
public setPaymentPeriod(\YooKassa\Request\Payments\PaymentPeriod|array|null $payment_period = null) : self
```

**Summary**

Устанавливает payment_period.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\PaymentPeriod OR array OR null</code> | payment_period  | Период оплаты, за который выставлены начисления и за который вносится оплата. |

**Returns:** self - 


<a name="method_setPaymentPurpose" class="anchor"></a>
#### public setPaymentPurpose() : self

```php
public setPaymentPurpose(string|array|null $payment_purpose = null) : self
```

**Summary**

Устанавливает payment_purpose.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR array OR null</code> | payment_purpose  | Назначение платежа (не больше 210 символов). |

**Returns:** self - 


<a name="method_setRecipient" class="anchor"></a>
#### public setRecipient() : self

```php
public setRecipient(\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient|array|null $recipient = null) : self
```

**Summary**

Устанавливает recipient.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient OR array OR null</code> | recipient  | Получатель платежа — государственная или коммерческая организация, которая предоставляет услуги или является информационным посредником, который собирает и обрабатывает начисления от других поставщиков услуг. |

**Returns:** self - 


<a name="method_setServiceId" class="anchor"></a>
#### public setServiceId() : self

```php
public setServiceId(string|null $service_id = null) : self
```

**Summary**

Устанавливает идентификатор жилищно-коммунальной услуги (ЖКУ).

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | service_id  | Идентификатор жилищно-коммунальной услуги (ЖКУ). Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `unified_account_number`. |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает type.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder](../classes/YooKassa-Request-Payments-PaymentOrderData-AbstractPaymentOrder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Код вида платежного поручения |

**Returns:** self - 


<a name="method_setUnifiedAccountNumber" class="anchor"></a>
#### public setUnifiedAccountNumber() : self

```php
public setUnifiedAccountNumber(string|null $unified_account_number = null) : self
```

**Summary**

Устанавливает единый лицевой счет.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | unified_account_number  | Единый лицевой счет. Уникальный идентификатор в ГИС ЖКХ, который характеризует связку «собственник-помещение». Обязательный параметр, если не передан `payment_document_id`, `payment_document_number`, `account_number` или `service_id`. |

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


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Проверяет, что все необходимые поля заданы.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderUtilities.md)

**Returns:** bool - 


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