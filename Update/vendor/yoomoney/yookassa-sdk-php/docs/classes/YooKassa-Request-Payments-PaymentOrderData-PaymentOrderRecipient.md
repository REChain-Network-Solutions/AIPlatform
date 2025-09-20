# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient
### Namespace: [\YooKassa\Request\Payments\PaymentOrderData](../namespaces/yookassa-request-payments-paymentorderdata.md)
---
**Summary:**

Класс, представляющий модель PaymentOrderRecipient.

**Description:**

Получатель платежа

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$bank](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#property_bank) |  | Банк получателя. |
| public | [$inn](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#property_inn) |  | ИНН получателя. |
| public | [$kpp](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#property_kpp) |  | КПП получателя. |
| public | [$name](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#property_name) |  | Название получателя. |

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
| public | [getBank()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_getBank) |  | Возвращает bank. |
| public | [getInn()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_getInn) |  | Возвращает inn. |
| public | [getKpp()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_getKpp) |  | Возвращает kpp. |
| public | [getName()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_getName) |  | Возвращает name. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setBank()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_setBank) |  | Устанавливает bank. |
| public | [setInn()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_setInn) |  | Устанавливает inn. |
| public | [setKpp()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_setKpp) |  | Устанавливает kpp. |
| public | [setName()](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md#method_setName) |  | Устанавливает name. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/PaymentOrderData/PaymentOrderRecipient.php](../../lib/Request/Payments/PaymentOrderData/PaymentOrderRecipient.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient

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
<a name="property_bank"></a>
#### public $bank : \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank
---
***Description***

Банк получателя.

**Type:** <a href="../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipientBank.html"><abbr title="\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank">PaymentOrderRecipientBank</abbr></a>

**Details:**


<a name="property_inn"></a>
#### public $inn : string
---
***Description***

ИНН получателя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_kpp"></a>
#### public $kpp : string
---
***Description***

КПП получателя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_name"></a>
#### public $name : string
---
***Description***

Название получателя.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

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


<a name="method_getBank" class="anchor"></a>
#### public getBank() : \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank|null

```php
public getBank() : \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank|null
```

**Summary**

Возвращает bank.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

**Returns:** \YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank|null - 


<a name="method_getInn" class="anchor"></a>
#### public getInn() : string|null

```php
public getInn() : string|null
```

**Summary**

Возвращает inn.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

**Returns:** string|null - 


<a name="method_getKpp" class="anchor"></a>
#### public getKpp() : string|null

```php
public getKpp() : string|null
```

**Summary**

Возвращает kpp.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

**Returns:** string|null - 


<a name="method_getName" class="anchor"></a>
#### public getName() : string|null

```php
public getName() : string|null
```

**Summary**

Возвращает name.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

**Returns:** string|null - 


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


<a name="method_setBank" class="anchor"></a>
#### public setBank() : self

```php
public setBank(\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank|array|null $bank = null) : self
```

**Summary**

Устанавливает bank.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank OR array OR null</code> | bank  | Банк получателя. |

**Returns:** self - 


<a name="method_setInn" class="anchor"></a>
#### public setInn() : self

```php
public setInn(string|null $inn = null) : self
```

**Summary**

Устанавливает inn.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | inn  | ИНН получателя. |

**Returns:** self - 


<a name="method_setKpp" class="anchor"></a>
#### public setKpp() : self

```php
public setKpp(string|null $kpp = null) : self
```

**Summary**

Устанавливает kpp.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | kpp  | КПП получателя. |

**Returns:** self - 


<a name="method_setName" class="anchor"></a>
#### public setName() : self

```php
public setName(string|null $name = null) : self
```

**Summary**

Устанавливает name.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient](../classes/YooKassa-Request-Payments-PaymentOrderData-PaymentOrderRecipient.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | name  | Название получателя. |

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