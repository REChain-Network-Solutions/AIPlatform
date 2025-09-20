# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Common\ListObject
### Namespace: [\YooKassa\Common](../namespaces/yookassa-common.md)
---
**Summary:**

Класс, представляющий модель ListObject.

**Description:**

Коллекция объектов AbstractObject.

---
### Examples
Работа с коллекциями объектов

```php
require_once '../vendor/autoload.php';

// Создание коллекции
$collection = new \YooKassa\Common\ListObject(\YooKassa\Request\Payments\Airline::class, [
    [
        'booking_reference' => 'IIIKRV',
        'ticket_number' => '12342123413',
        'passengers' => [
            [
                'first_name' => 'SERGEI',
                'last_name' => 'IVANOV',
            ],
        ],
        'legs' => [
            [
                'departure_airport' => 'LED',
                'destination_airport' => 'AMS',
                'departure_date' => '2023-01-20',
            ],
        ],
    ],
]);

var_dump([$collection->count(), $collection->toArray(), json_encode($collection->toArray())]);

// Чтобы сменить тип объекта коллекции, необходимо сначала очистить коллекцию. В случае попытки изменить тип объекта в заполненной коллекции, будет получено исключение
try {
    $collection->setType(\YooKassa\Request\Payments\Leg::class);
} catch (Exception $exception) {
    print_r($exception);
    exit();
}

// Очищает коллекцию и устанавливает тип объекта в коллекцию
$collection->clear()->setType(\YooKassa\Request\Payments\Passenger::class);

// Добавляет объект в коллекцию
$collection->add(new \YooKassa\Request\Payments\Passenger(['first_name' => 'Sergei', 'last_name' => 'Ivanov']));

var_dump([$collection->count(), $collection->toArray(), json_encode($collection->toArray())]);

// Получить массив объектов
var_dump($collection->getItems()->toArray());

// Для добавления объектов в коллекцию можно использовать []
$collection[] = ['first_name' => 'Ivan', 'last_name' => 'Ivanov'];

// Можно удалить объект из коллекции по индексу
$collection->remove(1);

var_dump([$collection->count(), $collection->toArray(), json_encode($collection->toArray())]);

// Можно добавить массив объектов в коллекцию. Если тип коллекции интерфейс или абстрактный класс, то добавить массив не получится.
$collection->merge([
    ['first_name' => 'Michail', 'last_name' => 'Sidorov'],
    new \YooKassa\Request\Payments\Passenger(['first_name' => 'Alex', 'last_name' => 'Lutor']),
]);

// Можно получить объект коллекции по индексу
var_dump([$collection[0]?->toArray(), $collection->get(1)?->toArray()]);

```

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-ListObject.md#method___construct) |  |  |
| public | [add()](../classes/YooKassa-Common-ListObject.md#method_add) |  | Добавляет объект в коллекцию |
| public | [clear()](../classes/YooKassa-Common-ListObject.md#method_clear) |  | Очищает коллекцию |
| public | [count()](../classes/YooKassa-Common-ListObject.md#method_count) |  | Возвращает количество объектов в коллекции |
| public | [get()](../classes/YooKassa-Common-ListObject.md#method_get) |  | Возвращает объект коллекции по индексу |
| public | [getItems()](../classes/YooKassa-Common-ListObject.md#method_getItems) |  | Возвращает коллекцию |
| public | [getIterator()](../classes/YooKassa-Common-ListObject.md#method_getIterator) |  |  |
| public | [getType()](../classes/YooKassa-Common-ListObject.md#method_getType) |  | Возвращает тип объектов в коллекции |
| public | [isEmpty()](../classes/YooKassa-Common-ListObject.md#method_isEmpty) |  | Проверка на пустую коллекцию |
| public | [jsonSerialize()](../classes/YooKassa-Common-ListObject.md#method_jsonSerialize) |  | Возвращает коллекцию в виде массива |
| public | [merge()](../classes/YooKassa-Common-ListObject.md#method_merge) |  | Добавляет массив объектов в коллекцию |
| public | [offsetExists()](../classes/YooKassa-Common-ListObject.md#method_offsetExists) |  |  |
| public | [offsetGet()](../classes/YooKassa-Common-ListObject.md#method_offsetGet) |  |  |
| public | [offsetSet()](../classes/YooKassa-Common-ListObject.md#method_offsetSet) |  |  |
| public | [offsetUnset()](../classes/YooKassa-Common-ListObject.md#method_offsetUnset) |  |  |
| public | [remove()](../classes/YooKassa-Common-ListObject.md#method_remove) |  | Удаляет объект из коллекции по индексу |
| public | [setType()](../classes/YooKassa-Common-ListObject.md#method_setType) |  | Устанавливает тип объектов в коллекции |
| public | [toArray()](../classes/YooKassa-Common-ListObject.md#method_toArray) |  | Возвращает коллекцию в виде массива |

---
### Details
* File: [lib/Common/ListObject.php](../../lib/Common/ListObject.php)
* Package: YooKassa
* Class Hierarchy:
  * \YooKassa\Common\ListObject
* Implements:
  * [\YooKassa\Common\ListObjectInterface](../classes/YooKassa-Common-ListObjectInterface.md)

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(string $type, array|null $data = []) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  | Тип хранимых объектов |
| <code lang="php">array OR null</code> | data  | Массив данных |

**Returns:** mixed - 


<a name="method_add" class="anchor"></a>
#### public add() : $this

```php
public add(mixed $item) : $this
```

**Summary**

Добавляет объект в коллекцию

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | item  |  |

**Returns:** $this - 


<a name="method_clear" class="anchor"></a>
#### public clear() : $this

```php
public clear() : $this
```

**Summary**

Очищает коллекцию

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** $this - 


<a name="method_count" class="anchor"></a>
#### public count() : int

```php
public count() : int
```

**Summary**

Возвращает количество объектов в коллекции

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** int - 


<a name="method_get" class="anchor"></a>
#### public get() : \YooKassa\Common\AbstractObject

```php
public get(int $index) : \YooKassa\Common\AbstractObject
```

**Summary**

Возвращает объект коллекции по индексу

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | index  |  |

**Returns:** \YooKassa\Common\AbstractObject - 


<a name="method_getItems" class="anchor"></a>
#### public getItems() : \Ds\Vector

```php
public getItems() : \Ds\Vector
```

**Summary**

Возвращает коллекцию

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** \Ds\Vector - 


<a name="method_getIterator" class="anchor"></a>
#### public getIterator() : \Ds\Vector

```php
public getIterator() : \Ds\Vector
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** \Ds\Vector - 


<a name="method_getType" class="anchor"></a>
#### public getType() : string

```php
public getType() : string
```

**Summary**

Возвращает тип объектов в коллекции

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** string - 


<a name="method_isEmpty" class="anchor"></a>
#### public isEmpty() : bool

```php
public isEmpty() : bool
```

**Summary**

Проверка на пустую коллекцию

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** bool - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает коллекцию в виде массива

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** array - 


<a name="method_merge" class="anchor"></a>
#### public merge() : $this

```php
public merge(iterable $data) : $this
```

**Summary**

Добавляет массив объектов в коллекцию

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">iterable</code> | data  |  |

**Returns:** $this - 


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : mixed

```php
public offsetExists(mixed $offset) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | offset  |  |

**Returns:** mixed - 


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(mixed $offset) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | offset  |  |

**Returns:** mixed - 


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : mixed

```php
public offsetSet(mixed $offset, mixed $value) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | offset  |  |
| <code lang="php">mixed</code> | value  |  |

**Returns:** mixed - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : mixed

```php
public offsetUnset(mixed $offset) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | offset  |  |

**Returns:** mixed - 


<a name="method_remove" class="anchor"></a>
#### public remove() : $this

```php
public remove(int $index) : $this
```

**Summary**

Удаляет объект из коллекции по индексу

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | index  |  |

**Returns:** $this - 


<a name="method_setType" class="anchor"></a>
#### public setType() : $this

```php
public setType(string $type) : $this
```

**Summary**

Устанавливает тип объектов в коллекции

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  |  |

**Returns:** $this - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает коллекцию в виде массива

**Details:**
* Inherited From: [\YooKassa\Common\ListObject](../classes/YooKassa-Common-ListObject.md)

**Returns:** array - 



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