# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PaymentMethods\PaymentMethodCard
### Namespace: [\YooKassa\Request\PaymentMethods](../namespaces/yookassa-request-paymentmethods.md)
---
**Summary:**

Класс, представляющий модель PaymentMethodCard.

**Description:**

Данные банковской карты (необходимы, если вы собираете данные карты пользователей на своей стороне).

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$cardholder](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#property_cardholder) |  | Имя владельца карты. |
| public | [$csc](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#property_csc) |  | Код CVC2 или CVV2, 3 или 4 символа, печатается на обратной стороне карты. |
| public | [$expiry_month](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#property_expiry_month) |  | Срок действия, месяц, MM. |
| public | [$expiry_year](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#property_expiry_year) |  | Срок действия, год, YYYY. |
| public | [$expiryMonth](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#property_expiryMonth) |  | Срок действия, месяц, MM. |
| public | [$expiryYear](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#property_expiryYear) |  | Срок действия, год, YYYY. |
| public | [$number](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#property_number) |  | Номер банковской карты. |

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
| public | [getCardholder()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_getCardholder) |  | Возвращает имя держателя карты. |
| public | [getCsc()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_getCsc) |  | Возвращает CVV2/CVC2 код. |
| public | [getExpiryMonth()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_getExpiryMonth) |  | Возвращает месяц срока действия карты. |
| public | [getExpiryYear()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_getExpiryYear) |  | Возвращает год срока действия карты. |
| public | [getNumber()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_getNumber) |  | Возвращает номер банковской карты. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCardholder()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_setCardholder) |  | Устанавливает имя держателя карты. |
| public | [setCsc()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_setCsc) |  | Устанавливает CVV2/CVC2 код. |
| public | [setExpiryMonth()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_setExpiryMonth) |  | Устанавливает месяц срока действия карты. |
| public | [setExpiryYear()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_setExpiryYear) |  | Устанавливает год срока действия карты. |
| public | [setNumber()](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md#method_setNumber) |  | Устанавливает номер банковской карты. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/PaymentMethods/PaymentMethodCard.php](../../lib/Request/PaymentMethods/PaymentMethodCard.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\PaymentMethods\PaymentMethodCard

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
<a name="property_cardholder"></a>
#### public $cardholder : string
---
***Description***

Имя владельца карты.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_csc"></a>
#### public $csc : string
---
***Description***

Код CVC2 или CVV2, 3 или 4 символа, печатается на обратной стороне карты.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiry_month"></a>
#### public $expiry_month : string
---
***Description***

Срок действия, месяц, MM.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiry_year"></a>
#### public $expiry_year : string
---
***Description***

Срок действия, год, YYYY.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiryMonth"></a>
#### public $expiryMonth : string
---
***Description***

Срок действия, месяц, MM.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expiryYear"></a>
#### public $expiryYear : string
---
***Description***

Срок действия, год, YYYY.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_number"></a>
#### public $number : string
---
***Description***

Номер банковской карты.

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


<a name="method_getCardholder" class="anchor"></a>
#### public getCardholder() : string|null

```php
public getCardholder() : string|null
```

**Summary**

Возвращает имя держателя карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

**Returns:** string|null - Имя держателя карты


<a name="method_getCsc" class="anchor"></a>
#### public getCsc() : string|null

```php
public getCsc() : string|null
```

**Summary**

Возвращает CVV2/CVC2 код.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

**Returns:** string|null - CVV2/CVC2 код


<a name="method_getExpiryMonth" class="anchor"></a>
#### public getExpiryMonth() : string|null

```php
public getExpiryMonth() : string|null
```

**Summary**

Возвращает месяц срока действия карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

**Returns:** string|null - Срок действия, месяц, MM


<a name="method_getExpiryYear" class="anchor"></a>
#### public getExpiryYear() : string|null

```php
public getExpiryYear() : string|null
```

**Summary**

Возвращает год срока действия карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

**Returns:** string|null - Срок действия, год, YYYY


<a name="method_getNumber" class="anchor"></a>
#### public getNumber() : string|null

```php
public getNumber() : string|null
```

**Summary**

Возвращает номер банковской карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

**Returns:** string|null - Номер банковской карты


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


<a name="method_setCardholder" class="anchor"></a>
#### public setCardholder() : self

```php
public setCardholder(string|null $cardholder = null) : self
```

**Summary**

Устанавливает имя держателя карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | cardholder  | Имя держателя карты |

**Returns:** self - 


<a name="method_setCsc" class="anchor"></a>
#### public setCsc() : self

```php
public setCsc(string|null $csc = null) : self
```

**Summary**

Устанавливает CVV2/CVC2 код.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | csc  | CVV2/CVC2 код |

**Returns:** self - 


<a name="method_setExpiryMonth" class="anchor"></a>
#### public setExpiryMonth() : self

```php
public setExpiryMonth(string|null $expiry_month = null) : self
```

**Summary**

Устанавливает месяц срока действия карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | expiry_month  | Срок действия, месяц, MM |

**Returns:** self - 


<a name="method_setExpiryYear" class="anchor"></a>
#### public setExpiryYear() : self

```php
public setExpiryYear(string|null $expiry_year = null) : self
```

**Summary**

Устанавливает год срока действия карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | expiry_year  | Срок действия, год, YYYY |

**Returns:** self - 


<a name="method_setNumber" class="anchor"></a>
#### public setNumber() : self

```php
public setNumber(string|null $number = null) : self
```

**Summary**

Устанавливает номер банковской карты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\PaymentMethodCard](../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | number  | Номер банковской карты |

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