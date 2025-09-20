# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Receipt\ReceiptCustomer
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Информация о плательщике.


---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$email](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#property_email) |  | E-mail адрес плательщика на который будет выслан чек. |
| public | [$full_name](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#property_full_name) |  | Для юрлица — название организации, для ИП и физического лица — ФИО. |
| public | [$fullName](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#property_fullName) |  | Для юрлица — название организации, для ИП и физического лица — ФИО. |
| public | [$inn](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#property_inn) |  | ИНН плательщика (10 или 12 цифр). |
| public | [$phone](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#property_phone) |  | Номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек. |

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
| public | [getEmail()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_getEmail) |  | Возвращает адрес электронной почты на который будет выслан чек. |
| public | [getFullName()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_getFullName) |  | Возвращает для юрлица — название организации, для ИП и физического лица — ФИО. |
| public | [getInn()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_getInn) |  | Возвращает ИНН плательщика. |
| public | [getPhone()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_getPhone) |  | Возвращает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [isEmpty()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_isEmpty) |  | Проверка на заполненность объекта. |
| public | [jsonSerialize()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setEmail()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_setEmail) |  | Устанавливает адрес электронной почты на который будет выслан чек. |
| public | [setFullName()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_setFullName) |  | Устанавливает Название организации или ФИО. |
| public | [setInn()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_setInn) |  | Устанавливает ИНН плательщика. |
| public | [setPhone()](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md#method_setPhone) |  | Устанавливает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Receipt/ReceiptCustomer.php](../../lib/Model/Receipt/ReceiptCustomer.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Receipt\ReceiptCustomer
* Implements:
  * [\YooKassa\Model\Receipt\ReceiptCustomerInterface](../classes/YooKassa-Model-Receipt-ReceiptCustomerInterface.md)

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
<a name="property_email"></a>
#### public $email : string
---
***Description***

E-mail адрес плательщика на который будет выслан чек.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_full_name"></a>
#### public $full_name : string
---
***Description***

Для юрлица — название организации, для ИП и физического лица — ФИО.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_fullName"></a>
#### public $fullName : string
---
***Description***

Для юрлица — название организации, для ИП и физического лица — ФИО.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_inn"></a>
#### public $inn : string
---
***Description***

ИНН плательщика (10 или 12 цифр).

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_phone"></a>
#### public $phone : string
---
***Description***

Номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.

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


<a name="method_getEmail" class="anchor"></a>
#### public getEmail() : string|null

```php
public getEmail() : string|null
```

**Summary**

Возвращает адрес электронной почты на который будет выслан чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

**Returns:** string|null - E-mail адрес плательщика


<a name="method_getFullName" class="anchor"></a>
#### public getFullName() : string|null

```php
public getFullName() : string|null
```

**Summary**

Возвращает для юрлица — название организации, для ИП и физического лица — ФИО.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

**Returns:** string|null - Название организации или ФИО


<a name="method_getInn" class="anchor"></a>
#### public getInn() : string|null

```php
public getInn() : string|null
```

**Summary**

Возвращает ИНН плательщика.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

**Returns:** string|null - 


<a name="method_getPhone" class="anchor"></a>
#### public getPhone() : string|null

```php
public getPhone() : string|null
```

**Summary**

Возвращает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

**Returns:** string|null - Номер телефона плательщика


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_isEmpty" class="anchor"></a>
#### public isEmpty() : bool

```php
public isEmpty() : bool
```

**Summary**

Проверка на заполненность объекта.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

**Returns:** bool - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

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


<a name="method_setEmail" class="anchor"></a>
#### public setEmail() : self

```php
public setEmail(string|null $email = null) : self
```

**Summary**

Устанавливает адрес электронной почты на который будет выслан чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | email  | E-mail адрес плательщика |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException | Выбрасывается, если в качестве значения была передана не строка |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueException | Выбрасывается если Email не соответствует формату |

**Returns:** self - 


<a name="method_setFullName" class="anchor"></a>
#### public setFullName() : self

```php
public setFullName(string|null $full_name = null) : self
```

**Summary**

Устанавливает Название организации или ФИО.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | full_name  | Название организации или ФИО |

**Returns:** self - 


<a name="method_setInn" class="anchor"></a>
#### public setInn() : self

```php
public setInn(string|null $inn = null) : self
```

**Summary**

Устанавливает ИНН плательщика.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | inn  | ИНН плательщика (10 или 12 цифр) |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException | Выбрасывается, если в качестве значения была передана не строка |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueException | Выбрасывается если ИНН не соответствует формату 10 или 12 цифр |

**Returns:** self - 


<a name="method_setPhone" class="anchor"></a>
#### public setPhone() : self

```php
public setPhone(string|null $phone = null) : self
```

**Summary**

Устанавливает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\ReceiptCustomer](../classes/YooKassa-Model-Receipt-ReceiptCustomer.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | phone  | Номер телефона плательщика в формате ITU-T E.164 |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException | Выбрасывается, если в качестве значения была передана не строка |

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