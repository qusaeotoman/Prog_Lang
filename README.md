1\) Задаём пути



$py  = "C:\\Users\\qusae\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"

$dll = (Resolve-Path .\\tree-sitter\\var1.dll).Path



\# Входной файл: первый файл из examples\_v1\\testv1

$in  = (Get-ChildItem .\\examples\_v1\\testv1 -File | Select-Object -First 1).FullName





2 (Очистка и запуск 



Remove-Item -Recurse -Force .\\out -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force .\\out | Out-Null



\& $py .\\python\\main.py --lib var1 $dll $in .\\out





Отладка (полезные команды)

1\) Показать все файлы дерева



Get-ChildItem .\\out\\tree -Recurse -File | Select-Object FullName



2\) Показать первые 120 строк первого дерева



Get-Content (Get-ChildItem .\\out\\tree -Recurse -File | Select-Object -First 1).FullName -TotalCount 120



3\) Поиск функции в дереве (пример: writeInt)



$tree = (Get-ChildItem .\\out\\tree -Recurse -File | Select-Object -First 1).FullName

Select-String -Path $tree -Pattern '"writeInt"' -Context 0,60



