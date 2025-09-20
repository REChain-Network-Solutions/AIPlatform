<?php

/**
 * The MIT License
 *
 * Copyright (c) 2024 "YooMoney", NBСO LLC
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

namespace YooKassa\Validator;

class Validator
{
    private array $propRules = [];
    private array $propValues = [];
    private object $object;

    public function __construct(object $object)
    {
        $this->object = $object;

        $this->parsePropRules();
    }

    /**
     * @param string $propertyName
     * @param mixed $propertyValue
     * @throws \InvalidArgumentException
     * @return void
     */
    public function validatePropertyValue(string $propertyName, mixed $propertyValue, ?array $filter = []): void
    {
        if (!isset($this->propRules[$propertyName])) {
            return;
        }

        foreach ($this->propRules[$propertyName] as $constraint) {
            if (!empty($filter) && in_array($constraint::class, $filter)) {
                continue;
            }
            $validator = $constraint->validatedBy();
            $validator = new $validator($this->object::class, $propertyName);
            $validator->validate($propertyValue, $constraint);
        }
    }

    /**
     * @return void
     * @throws \InvalidArgumentException
     */
    public function validateAllProperties(): void
    {
        foreach($this->propValues as $propName => $propValue) {
            $this->validatePropertyValue($propName, $propValue);
        }
    }

    /**
     * @param string $propName
     * @return array|null
     */
    public function getRulesByPropName(string $propName): ?array
    {
        return $this->propRules[$propName] ?? null;
    }

    /**
     * @return void
     */
    private function parsePropRules(): void
    {
        $reflector = new \ReflectionObject($this->object);
        foreach ($reflector->getProperties() as $property) {
            $property->setAccessible(true);
            if ($property->isInitialized($this->object)) {
                $this->propValues[$property->getName()] = $property->getValue($this->object);
            }
            foreach ($property->getAttributes() as $attribute) {
                $this->propRules[$property->getName()][] = $attribute->newInstance();
            }
        }
    }
}
