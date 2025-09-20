# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate
### Namespace: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate](../namespaces/yookassa-model-payment-paymentmethod-electroniccertificate.md)
---
**Summary:**

Класс, представляющий модель ElectronicCertificate.

**Description:**

Описание используемого электронного сертификата.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$applied_compensation](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_applied_compensation) |  | Сумма, которую одобрили для оплаты по сертификату за одну единицу товара. Пример: из 1000 рублей одобрили 500 рублей для оплаты по сертификату. |
| public | [$appliedCompensation](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_appliedCompensation) |  | Сумма, которую одобрили для оплаты по сертификату за одну единицу товара. Пример: из 1000 рублей одобрили 500 рублей для оплаты по сертификату. |
| public | [$available_compensation](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_available_compensation) |  | Максимально допустимая сумма, которую может покрыть электронный сертификат для оплаты одной единицы товара. Пример: сертификат может компенсировать максимум 1000 рублей для оплаты этого товара. |
| public | [$availableCompensation](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_availableCompensation) |  | Максимально допустимая сумма, которую может покрыть электронный сертификат для оплаты одной единицы товара. Пример: сертификат может компенсировать максимум 1000 рублей для оплаты этого товара. |
| public | [$certificate_id](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_certificate_id) |  | Идентификатор сертификата. От 20 до 30 символов. |
| public | [$certificateId](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_certificateId) |  | Идентификатор сертификата. От 20 до 30 символов. |
| public | [$tru_quantity](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_tru_quantity) |  | Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату. |
| public | [$truQuantity](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#property_truQuantity) |  | Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAppliedCompensation()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_getAppliedCompensation) |  | Возвращает applied_compensation. |
| public | [getAvailableCompensation()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_getAvailableCompensation) |  | Возвращает available_compensation. |
| public | [getCertificateId()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_getCertificateId) |  | Возвращает certificate_id. |
| public | [getTruQuantity()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_getTruQuantity) |  | Возвращает tru_quantity. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAppliedCompensation()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_setAppliedCompensation) |  | Устанавливает applied_compensation. |
| public | [setAvailableCompensation()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_setAvailableCompensation) |  | Устанавливает available_compensation. |
| public | [setCertificateId()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_setCertificateId) |  | Устанавливает certificate_id. |
| public | [setTruQuantity()](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md#method_setTruQuantity) |  | Устанавливает tru_quantity. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Payment/PaymentMethod/ElectronicCertificate/ElectronicCertificate.php](../../lib/Model/Payment/PaymentMethod/ElectronicCertificate/ElectronicCertificate.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate

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
<a name="property_applied_compensation"></a>
#### public $applied_compensation : \YooKassa\Model\AmountInterface
---
***Description***

Сумма, которую одобрили для оплаты по сертификату за одну единицу товара. Пример: из 1000 рублей одобрили 500 рублей для оплаты по сертификату.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_appliedCompensation"></a>
#### public $appliedCompensation : \YooKassa\Model\AmountInterface
---
***Description***

Сумма, которую одобрили для оплаты по сертификату за одну единицу товара. Пример: из 1000 рублей одобрили 500 рублей для оплаты по сертификату.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_available_compensation"></a>
#### public $available_compensation : \YooKassa\Model\AmountInterface
---
***Description***

Максимально допустимая сумма, которую может покрыть электронный сертификат для оплаты одной единицы товара. Пример: сертификат может компенсировать максимум 1000 рублей для оплаты этого товара.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_availableCompensation"></a>
#### public $availableCompensation : \YooKassa\Model\AmountInterface
---
***Description***

Максимально допустимая сумма, которую может покрыть электронный сертификат для оплаты одной единицы товара. Пример: сертификат может компенсировать максимум 1000 рублей для оплаты этого товара.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_certificate_id"></a>
#### public $certificate_id : string
---
***Description***

Идентификатор сертификата. От 20 до 30 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_certificateId"></a>
#### public $certificateId : string
---
***Description***

Идентификатор сертификата. От 20 до 30 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_tru_quantity"></a>
#### public $tru_quantity : int
---
***Description***

Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_truQuantity"></a>
#### public $truQuantity : int
---
***Description***

Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату.

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


<a name="method_getAppliedCompensation" class="anchor"></a>
#### public getAppliedCompensation() : \YooKassa\Model\AmountInterface|null

```php
public getAppliedCompensation() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает applied_compensation.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма, которую одобрили для оплаты по сертификату


<a name="method_getAvailableCompensation" class="anchor"></a>
#### public getAvailableCompensation() : \YooKassa\Model\AmountInterface|null

```php
public getAvailableCompensation() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает available_compensation.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Максимально допустимая сумма


<a name="method_getCertificateId" class="anchor"></a>
#### public getCertificateId() : string|null

```php
public getCertificateId() : string|null
```

**Summary**

Возвращает certificate_id.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

**Returns:** string|null - Идентификатор сертификата


<a name="method_getTruQuantity" class="anchor"></a>
#### public getTruQuantity() : int|null

```php
public getTruQuantity() : int|null
```

**Summary**

Возвращает tru_quantity.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

**Returns:** int|null - Количество единиц товара


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


<a name="method_setAppliedCompensation" class="anchor"></a>
#### public setAppliedCompensation() : self

```php
public setAppliedCompensation(\YooKassa\Model\AmountInterface|array|null $applied_compensation = null) : self
```

**Summary**

Устанавливает applied_compensation.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | applied_compensation  | Сумма, которую одобрили для оплаты по сертификату |

**Returns:** self - 


<a name="method_setAvailableCompensation" class="anchor"></a>
#### public setAvailableCompensation() : self

```php
public setAvailableCompensation(\YooKassa\Model\AmountInterface|array|null $available_compensation = null) : self
```

**Summary**

Устанавливает available_compensation.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | available_compensation  | Максимально допустимая сумма |

**Returns:** self - 


<a name="method_setCertificateId" class="anchor"></a>
#### public setCertificateId() : self

```php
public setCertificateId(string|null $certificate_id = null) : self
```

**Summary**

Устанавливает certificate_id.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | certificate_id  | Идентификатор сертификата. От 20 до 30 символов. |

**Returns:** self - 


<a name="method_setTruQuantity" class="anchor"></a>
#### public setTruQuantity() : self

```php
public setTruQuantity(int|null $tru_quantity = null) : self
```

**Summary**

Устанавливает tru_quantity.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificate](../classes/YooKassa-Model-Payment-PaymentMethod-ElectronicCertificate-ElectronicCertificate.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | tru_quantity  | Количество единиц товара, которое одобрили для оплаты по этому электронному сертификату. |

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