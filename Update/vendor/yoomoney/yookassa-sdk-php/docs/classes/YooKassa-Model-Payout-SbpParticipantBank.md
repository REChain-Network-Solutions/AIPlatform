# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payout\SbpParticipantBank
### Namespace: [\YooKassa\Model\Payout](../namespaces/yookassa-model-payout.md)
---
**Summary:**

Класс, представляющий модель SbpParticipantBank.

**Description:**

Объект участника СБП (Системы быстрых платежей ЦБ РФ) — актуальная информация о банке или платежном сервисе, подключенном к СБП.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_BANK_ID](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#constant_MAX_LENGTH_BANK_ID) |  |  |
| public | [MAX_LENGTH_NAME](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#constant_MAX_LENGTH_NAME) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$bank_id](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#property_bank_id) |  | Идентификатор банка или платежного сервиса в СБП. |
| public | [$bankId](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#property_bankId) |  | Идентификатор банка или платежного сервиса в СБП. |
| public | [$bic](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#property_bic) |  | Банковский идентификационный код (БИК) банка или платежного сервиса. |
| public | [$name](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#property_name) |  | Название банка или платежного сервиса в СБП. |

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
| public | [getBankId()](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#method_getBankId) |  | Возвращает идентификатор банка или платежного сервиса в СБП. |
| public | [getBic()](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#method_getBic) |  | Возвращает банковский идентификационный код. |
| public | [getName()](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#method_getName) |  | Возвращает название банка или платежного сервиса в СБП. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setBankId()](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#method_setBankId) |  | Устанавливает идентификатор банка или платежного сервиса в СБП. |
| public | [setBic()](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#method_setBic) |  | Устанавливает банковский идентификационный код. |
| public | [setName()](../classes/YooKassa-Model-Payout-SbpParticipantBank.md#method_setName) |  | Устанавливает название банка или платежного сервиса в СБП. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Payout/SbpParticipantBank.php](../../lib/Model/Payout/SbpParticipantBank.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Payout\SbpParticipantBank

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
<a name="constant_MAX_LENGTH_BANK_ID" class="anchor"></a>
###### MAX_LENGTH_BANK_ID
```php
MAX_LENGTH_BANK_ID = 12 : int
```


<a name="constant_MAX_LENGTH_NAME" class="anchor"></a>
###### MAX_LENGTH_NAME
```php
MAX_LENGTH_NAME = 128 : int
```



---
## Properties
<a name="property_bank_id"></a>
#### public $bank_id : string
---
***Description***

Идентификатор банка или платежного сервиса в СБП.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_bankId"></a>
#### public $bankId : string
---
***Description***

Идентификатор банка или платежного сервиса в СБП.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_bic"></a>
#### public $bic : string
---
***Description***

Банковский идентификационный код (БИК) банка или платежного сервиса.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_name"></a>
#### public $name : string
---
***Description***

Название банка или платежного сервиса в СБП.

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


<a name="method_getBankId" class="anchor"></a>
#### public getBankId() : string|null

```php
public getBankId() : string|null
```

**Summary**

Возвращает идентификатор банка или платежного сервиса в СБП.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\SbpParticipantBank](../classes/YooKassa-Model-Payout-SbpParticipantBank.md)

**Returns:** string|null - Идентификатор банка или платежного сервиса в СБП


<a name="method_getBic" class="anchor"></a>
#### public getBic() : string|null

```php
public getBic() : string|null
```

**Summary**

Возвращает банковский идентификационный код.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\SbpParticipantBank](../classes/YooKassa-Model-Payout-SbpParticipantBank.md)

**Returns:** string|null - Банковский идентификационный код (БИК) банка или платежного сервиса


<a name="method_getName" class="anchor"></a>
#### public getName() : string|null

```php
public getName() : string|null
```

**Summary**

Возвращает название банка или платежного сервиса в СБП.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\SbpParticipantBank](../classes/YooKassa-Model-Payout-SbpParticipantBank.md)

**Returns:** string|null - Название банка или платежного сервиса в СБП


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


<a name="method_setBankId" class="anchor"></a>
#### public setBankId() : self

```php
public setBankId(string|null $bank_id = null) : self
```

**Summary**

Устанавливает идентификатор банка или платежного сервиса в СБП.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\SbpParticipantBank](../classes/YooKassa-Model-Payout-SbpParticipantBank.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | bank_id  | Идентификатор банка или платежного сервиса в СБП |

**Returns:** self - 


<a name="method_setBic" class="anchor"></a>
#### public setBic() : self

```php
public setBic(string|null $bic = null) : self
```

**Summary**

Устанавливает банковский идентификационный код.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\SbpParticipantBank](../classes/YooKassa-Model-Payout-SbpParticipantBank.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | bic  | Банковский идентификационный код (БИК) банка или платежного сервиса |

**Returns:** self - 


<a name="method_setName" class="anchor"></a>
#### public setName() : self

```php
public setName(string|null $name = null) : self
```

**Summary**

Устанавливает название банка или платежного сервиса в СБП.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\SbpParticipantBank](../classes/YooKassa-Model-Payout-SbpParticipantBank.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | name  | Название банка или платежного сервиса в СБП |

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